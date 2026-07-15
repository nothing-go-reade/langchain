# LangChain源码解析07：BaseChatModel如何统一模型调用 发布包

## 公众号标题

LangChain源码解析07：BaseChatModel如何统一模型调用

建议系列名：LangChain 源码通关：从 Runnable 到 Agent 的工程内核

## 摘要

第七篇聚焦 `BaseChatModel`：解释它如何把宽输入归一化为 `PromptValue`，如何通过 `invoke -> generate_prompt -> generate -> _generate_with_cache` 串起单次调用、批量调用、callback/tracing、cache、rate limiter、stream fallback、provider hook、`bind_tools` 和 `with_structured_output`。

## 文件清单

- HTML 粘贴版：`docs/wechat/langchain-source-07/langchain-source-07-wechat.html`
- WeChat Markdown：`docs/wechat/langchain-source-07/langchain-source-07-wechat.md`
- 源稿 Markdown：`docs/wechat/langchain-source-07/langchain-source-07.md`
- 调用主线图：`docs/wechat/langchain-source-07/assets/base-chat-model-pipeline.png`
- cache/stream 决策图：`docs/wechat/langchain-source-07/assets/base-chat-model-cache-stream.png`

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

- 第 1 篇：`LangChain源码解析01：先看懂Agent工程骨架`，https://mp.weixin.qq.com/s/tPhQNpcwcDNPmNTfealwhA
- 第 2 篇：`LangChain源码解析02：Runnable把一切串起来`，https://mp.weixin.qq.com/s/cOYJN_7pZ3FZbVRdAD95ww
- 第 3 篇：`LangChain源码解析03：RunnableConfig如何追踪到底`，https://mp.weixin.qq.com/s/u7WqvJhNkjUW-LCzWNyhLQ
- 第 4 篇：`LangChain源码解析04：Message不只是字符串`，https://mp.weixin.qq.com/s/IoS6e0hHx9uuhegH6WvAxA
- 第 5 篇：`LangChain源码解析05：Tool如何从函数变成契约`，https://mp.weixin.qq.com/s/RdojltI3OiONkSsG0rTTaA
- 第 6 篇：`LangChain源码解析06：Prompt和Parser守住两端`，https://mp.weixin.qq.com/s/qKk6xfZRkSCpBeQlEHBrAA
- 第 7 篇：`LangChain源码解析07：BaseChatModel如何统一模型调用`，当前篇，https://mp.weixin.qq.com/s/hHbN-NPmvdDAPLjsscdWCA

## 公开链接

https://mp.weixin.qq.com/s/hHbN-NPmvdDAPLjsscdWCA
## 手工检查

- HTML 应内嵌 2 张 PNG base64，可直接复制到公众号编辑器。
- WeChat Markdown 不应包含 Mermaid。
- 图表均有 PNG 文件。
- 公众号正文不应出现本地路径、AI 过程描述或生成流程描述。
- 标题需要在公众号编辑器确认计数不超过 64/64。

## 源码参考

GitHub: https://github.com/langchain-ai/langchain
