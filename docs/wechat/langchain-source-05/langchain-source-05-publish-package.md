# LangChain源码解析05：Tool如何从函数变成契约 发布包

## 公众号标题

LangChain源码解析05：Tool如何从函数变成契约

建议系列名：LangChain 源码通关：从 Runnable 到 Agent 的工程内核

## 摘要

第五篇聚焦 LangChain Tool 体系：解释 `@tool` 如何把普通函数转换成 `StructuredTool`，`create_schema_from_function` 如何从函数签名、类型注解和 docstring 推导 Pydantic schema，`tool_call_schema` 如何过滤运行时注入参数，`InjectedToolArg` / `InjectedToolCallId` / `ToolRuntime` 如何建立模型可见参数和系统上下文之间的边界，以及工具执行结果如何包装成 `ToolMessage` 回填到消息列表。

## 文件清单

- HTML 粘贴版：`docs/wechat/langchain-source-05/langchain-source-05-wechat.html`
- WeChat Markdown：`docs/wechat/langchain-source-05/langchain-source-05-wechat.md`
- 源稿 Markdown：`docs/wechat/langchain-source-05/langchain-source-05.md`
- Tool schema 主链路图：`docs/wechat/langchain-source-05/assets/tool-schema-pipeline.png`
- 注入参数边界图：`docs/wechat/langchain-source-05/assets/tool-injected-boundary.png`

## 系列规划

建议用 24 篇讲通透，每天 1 篇，大约 24 天。

1. 总地图：LangChain 的分层、主线和阅读方法
2. Runnable：invoke、stream、batch 与 LCEL 组合
3. RunnableConfig 与 callback/tracing 传递
4. Message 体系：Human/AI/Tool 与 content blocks
5. Tool 体系：schema 推导、Injected 参数、ToolMessage
6. Prompt 与 output parser：链路两端的数据形状
7. BaseChatModel：generate、stream、cache、rate limiter
8. init_chat_model：provider registry 与可配置模型
9. create_agent：StateGraph 编译主流程
10. Agent 条件边：model/tools/end 的路由判断
11. middleware 总览：before/after/wrap hooks
12. structured output：ToolStrategy 与 ProviderStrategy
13. tool runtime：state/store/runtime 注入机制
14. streaming：chunk、event、transformer 与生命周期
15. LangGraph 集成：为什么 Agent 变成图
16. classic 对照：Chain 与 AgentExecutor 的旧范式
17. OpenAI integration：Chat Completions 与 Responses API
18. Anthropic integration：content blocks、tools、thinking
19. standard-tests：集成包的契约治理
20. text-splitters：DocumentTransformer 与切分策略
21. model-profiles：模型能力数据如何进入运行时
22. tracing/LangSmith：可观测性如何贯穿调用链
23. 扩展一个新 provider：接口、测试、CI 清单
24. 总结：LangChain 的工程方法论与可复用设计

## 系列链接

- 第 4 篇：`LangChain源码解析04：Message不只是字符串`，待发布 URL 后补齐。
- 第 3 篇：`LangChain源码解析03：RunnableConfig如何追踪到底`，待发布 URL 后补齐。
- 第 2 篇：`LangChain源码解析02：Runnable把一切串起来`，https://mp.weixin.qq.com/s/cOYJN_7pZ3FZbVRdAD95ww
- 第 1 篇：`LangChain源码解析01：先看懂Agent工程骨架`，待发布 URL 后补齐。
- 当前篇：第 5 篇。

## 手工检查

- HTML 应内嵌 2 张 PNG base64，可直接复制到公众号编辑器。
- WeChat Markdown 不应包含 Mermaid。
- 图表均有 PNG 文件。
- 公众号正文不应出现本地路径、AI 过程描述或生成流程描述。
- 标题需要在公众号编辑器确认计数不超过 64/64。

## 源码参考

GitHub: https://github.com/langchain-ai/langchain
