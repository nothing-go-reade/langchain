# LangChain源码解析02：Runnable把一切串起来

第二篇进入地基层：为什么 prompt、model、parser、tool 都能被同一种方式调用、组合、并追踪。

上一篇把 LangChain 的总地图搭起来之后，第二篇就该进入真正的地基：`Runnable`。

很多人第一次看到 LCEL，会先记住这句代码：

```python
chain = prompt | model | parser
```

这行代码确实漂亮，但如果只把它理解成“语法糖”，就低估了 LangChain 的设计重心。`|` 背后不是简单函数拼接，而是一套统一执行协议：单次调用、异步调用、批处理、流式输出、配置传递、callback 追踪，都要在同一个对象模型里成立。

所以这一篇的核心结论很简单：LangChain 不是先有一堆 chain，再把它们包装成 `Runnable`；它是先把“一段可执行工作”抽象成 `Runnable`，再让 prompt、model、parser、tool、retriever 都能被放进同一条工程链路。

![图 1：Runnable 把调用、批处理、流式和运行时上下文收进同一套协议](assets/runnable-protocol-map.png)

*图 1：Runnable 把调用、批处理、流式和运行时上下文收进同一套协议*


## 一、Runnable 先统一“怎么运行”

从源码结构看，`Runnable` 的第一层价值不是组合，而是统一运行入口。一个对象只要是 `Runnable`，它就至少要回答这些问题：

- 单个输入怎么变成输出：`invoke` / `ainvoke`。
- 多个输入怎么一起处理：`batch` / `abatch`。
- 输出能不能一点点吐出来：`stream` / `astream`。
- 上游如果也是流，下游能不能边收边处理：`transform` / `atransform`。
- 运行过程中怎么带上 tags、metadata、callbacks、configurable 这些上下文。

这里有个容易忽略的细节：基类给了很多默认实现，但默认实现不等于高性能实现。默认 `ainvoke` 会把同步 `invoke` 放进 executor；默认 `batch` 会并发调用多个 `invoke`；默认 `stream` 只是 `yield self.invoke(...)`。

> 也就是说，一个对象“能调用 `stream`”和“真的能边生成边输出”不是一回事。真正的流式能力，要看它有没有覆盖 `stream`，更要看它在组合链路里有没有实现 `transform`。

这就是 LangChain 工程味很重的地方：它先给所有对象一个保底协议，让上层能统一调度；再允许具体模型、检索器、解析器覆盖默认行为，把能力做到更深。


## 二、`|` 不是拼接，是生成 RunnableSequence

LCEL 里最常见的写法是：

```python
chain = prompt | model | parser
result = chain.invoke({"topic": "LangChain"})
```

从实现看，`__or__` 和 `pipe()` 最终都会创建 `RunnableSequence`。它的语义很朴素：前一个步骤的输出，变成后一个步骤的输入。但它做的事情不止这个。

`RunnableSequence.invoke` 会先启动一个根 run，然后每经过一个步骤，就用类似 `seq:step:n` 的名字创建子 callback。这样一条链路不是黑盒，而是一棵可追踪的运行树。

`batch` 也不是粗暴地对每个输入跑完整条链。它会按步骤推进：第一层先处理所有输入，第二层再处理第一层的所有输出。这样如果某个组件有自己的批处理优化，组合之后仍然能保留下来。

这也是 `RunnableSequence` 比普通函数组合更有价值的原因：它不仅表达顺序，还保留了批处理、异步、流式和 tracing 的结构信息。


## 三、字典为什么会变成并发分支

LCEL 里另一个很有意思的写法，是在链中间放一个 dict：

```python
chain = retriever | {
    "answer_context": format_docs,
    "raw_documents": lambda docs: docs,
}
```

这个 dict 会被规整成 `RunnableParallel`。它的语义是：把同一份输入同时交给多个分支，最后把每个分支的结果收成一个 dict。

从运行行为看，`RunnableParallel.invoke` 会复制当前 steps，使用 executor 并发提交每个分支，并给每个分支创建类似 `map:key:<key>` 的子 callback。也就是说，分支并发不是“写法上像并发”，而是运行时真的按分支组织。

这解释了为什么 LangChain 里经常可以自然地写出“保留原始结果 + 同时生成派生字段”的链路。你写的是 dict，运行时拿到的是并发可追踪的结构。

![图 2：LCEL 把表达式规整成 Sequence、Parallel、Lambda 和 Generator](assets/lcel-composition-flow.png)

*图 2：LCEL 把表达式规整成 Sequence、Parallel、Lambda 和 Generator*


## 四、coerce_to_runnable 是 LCEL 的入口闸门

为什么 prompt 后面可以接 model，model 后面可以接 parser，甚至还能接一个普通函数？关键在 `coerce_to_runnable`。它负责把“看起来像 Runnable 的东西”规整成真正的 `Runnable`：

- 已经是 `Runnable`：原样返回。
- 普通 callable：包装成 `RunnableLambda`。
- generator function：包装成 `RunnableGenerator`。
- dict：包装成 `RunnableParallel`。
- 其他类型：直接抛出类型错误。

这一步让 LCEL 的表达能力变得很宽。你不用手动把每个小函数都写成类，只要它符合可调用形态，就能被吸进同一套执行协议。

但这里也埋着一个重要边界：方便不代表没有代价。普通函数会变成 `RunnableLambda`，它很适合做非流式的小变换；如果这段逻辑要保留上游 chunk 的增量输出，就应该用 generator，或者自己实现 `transform`。


## 五、真正的流式，卡在 transform

很多 LangChain 链路的“卡顿感”，不是模型不支持流式，而是链中间某个步骤把流式截住了。

`RunnableSequence` 的文档和实现都强调同一件事：只要所有组件都实现了 `transform`，sequence 就能把上游流式输入一路传到下游；如果某个组件没有真正的 `transform`，流式就会从这个组件之后才开始。多个阻塞组件存在时，要等最后一个阻塞组件结束。

`RunnableLambda` 就是典型例子。它把普通 callable 接入协议，非常顺手，但默认不适合作为流式转换器。它通常需要先拿到完整输入，再产出结果。

`RunnableGenerator` 则是另一种取舍。它接受 `Iterator[A] -> Iterator[B]` 或异步版本，可以在上游 chunk 到来时就处理并吐出下游 chunk。对于自定义 parser、token 级别清洗、增量格式化，这类对象更合适。

> 一个实用判断：如果你的逻辑是“拿完整文本再算一次”，用 `RunnableLambda` 很自然；如果你的逻辑是“每来一段就处理一段”，就要优先考虑 `RunnableGenerator` 或自定义 `transform`。


## 六、RunnableConfig 让链路有运行时上下文

`Runnable` 还有一个容易被低估的部分：每个核心方法都接受 `config`。它不是简单的参数袋，而是整条链路的运行时上下文。

`ensure_config` 会补齐默认字段：`tags`、`metadata`、`callbacks`、`recursion_limit`、`configurable`。如果传入了一些不属于标准 config key 的字段，还会被放进 `configurable`。这就是为什么运行时切换模型、传递业务参数、补充 tracing 信息可以沿着 Runnable 链路往下走。

`patch_config` 则负责在进入子步骤时改写上下文，比如替换 callbacks、调整 recursion limit、设置 run name。配合上下文变量，父级配置能传给子 Runnable；配合 callback manager，运行轨迹能形成父子结构。

所以 `RunnableConfig` 和 `Callback` 不是额外装饰，而是 LangChain 能把复杂链路解释清楚的关键。没有它，`prompt | model | parser` 只能得到一个结果；有了它，才能知道每一步怎么运行、哪里耗时、哪里报错、哪个分支产生了哪个输出。


## 七、读 Runnable 源码的正确姿势

读这一层源码时，不要只盯着某一个方法。更有效的方式，是按四个问题来读：

1. 这个对象的输入输出是什么？有没有 schema？
2. 它是原生 `Runnable`，还是被 `coerce_to_runnable` 包装出来的？
3. 它在 sequence 或 parallel 里会生成什么运行结构？
4. 它有没有真正实现 `transform`，还是会在流式链路里形成缓冲点？

这四个问题能帮你快速判断一段 LangChain 代码到底是“可组合的工程对象”，还是只是“被临时包进链路的小函数”。


## 八、第二篇的结论

如果第一篇的结论是：LangChain 的主线是一套标准对象和图运行时。那么第二篇可以再往下压一句：`Runnable` 是这些标准对象共同遵守的执行协议。

它把一段 LLM 应用逻辑变成可调用、可批处理、可流式、可组合、可追踪、可配置的工程对象。LCEL 的简洁，来自这层协议的复杂；上层体验越轻，底层约束就越重。

理解了 `Runnable`，再去看 Prompt、Model、Tool、Parser、Agent，就不会觉得它们是散落的 API，而会看到同一个设计问题的不同答案：如何让不稳定的 LLM 调用，进入稳定的软件工程结构。


## 系列位置

当前文章：第 2 篇，拆 `Runnable` 与 LCEL 的执行地基。

历史文章：第 1 篇，`LangChain源码解析01：先看懂Agent工程骨架`。发布链接待补齐后会统一加入系列索引。

源码参考：
GitHub: https://github.com/langchain-ai/langchain

如果一次调用里同时有 prompt 渲染、模型流式、parser 增量解析和 callback 追踪，LangChain 怎么保证每个 chunk、每个子运行、每个配置项都能被正确接住？

---

## 源码审查笔记

写作基于 `1.3.11` 分支下的 LangChain Python monorepo，重点阅读：

- `libs/core/langchain_core/runnables/base.py`
  - `Runnable`：统一 `invoke`、`ainvoke`、`batch`、`stream`、`transform` 等执行入口。
  - `Runnable.__or__` / `Runnable.pipe`：组合为 `RunnableSequence`。
  - `RunnableSequence.invoke`：创建 root run，并为每个步骤 patch 子 callback，命名形如 `seq:step:n`。
  - `RunnableSequence.batch`：逐步骤调用组件自己的 `batch`，保留批处理优化。
  - `RunnableSequence.transform` / `stream`：通过每个步骤的 `transform` 串接流式能力。
  - `RunnableParallel.invoke`：同一输入进入多个分支，分支 callback 命名形如 `map:key:<key>`。
  - `RunnableLambda`：普通 callable 的包装，适合非流式小逻辑；返回 `Runnable` 时会递归调用。
  - `RunnableGenerator`：生成器函数包装，适合保留 chunk 级流式处理。
  - `coerce_to_runnable`：把 Runnable、callable、generator、dict 规整为标准 Runnable。
- `libs/core/langchain_core/runnables/config.py`
  - `ensure_config`：补齐默认 config，并把未知 key 放入 `configurable`。
  - `patch_config`：为子步骤替换 callbacks、run_name、recursion_limit 等。
  - `set_config_context`：通过 ContextVar 和 tracing context 传播子 Runnable 配置。

文章公开版刻意不放长路径和源码行号，避免公众号正文过重；这些细节保留在源稿笔记中，方便后续回查。
