const MOCK_DATA = {
    // Simulated script. Time is in relative seconds from start.
    transcript: [
        { time: 1, speaker: "Speaker A", text: "好的，大家下午好。我们开始今天的技术评审会。" },
        { time: 4, speaker: "Speaker A", text: "今天的主题是关于下一代实时语音转写系统的架构设计。主要内容在文档里已经发给大家了。" },
        { time: 9, speaker: "Speaker B", text: "收到。我看了一下文档，主要关注的是实时性这块。" },
        { time: 12, speaker: "Speaker B", text: "现在的延迟大概在多少？我注意到设计目标里写的是Draft状态小于1秒？" },
        { time: 16, speaker: "Speaker A", text: "对的。目前的架构原型，我们在端侧做了VAD和流式ASR的优化。" },
        { time: 20, speaker: "Speaker A", text: "草稿态（Draft）基本上都在500毫秒以内就能出来，体验非常跟手。" },
        { time: 24, speaker: "Speaker A", text: "如果是Final状态，因为需要回溯和标点预测，稍微慢一点，大概2秒左右。" },
        { time: 30, speaker: "Speaker C", text: "那关于那个“智能整理”的功能呢？它是怎么触发的？" },
        { time: 34, speaker: "Speaker A", text: "这个是系统的核心亮点。我们设定了一个3分钟的滑动窗口。" },
        { time: 38, speaker: "Speaker A", text: "每隔3分钟，后台的LLM会基于最近的转写内容，生成摘要和提纲，并提取出待办事项。" },
        { time: 45, speaker: "Speaker C", text: "如果3分钟内没什么新内容，也会强制更新吗？" },
        { time: 48, speaker: "Speaker A", text: "不会。我们有一个更新策略，会判断增量文本的Token数量和语义相似度。如果变化不大，就跳过更新，避免打扰用户。" },
        { time: 55, speaker: "Speaker B", text: "这块设计挺好的。我有另外一个担心，关于隐私问题。" },
        { time: 58, speaker: "Speaker B", text: "这些处理是在本地还是云端？如果上传云端，合规性怎么保证？" },
        { time: 63, speaker: "Speaker A", text: "这是一个关键点（Key Point）。我们的设计原则是“本地优先”。" },
        { time: 66, speaker: "Speaker A", text: "转写和基础的VAD都在本地运行。对于智能整理，如果设备性能足够（比如有NPU），也优先本地处理。" },
        { time: 73, speaker: "Speaker A", text: "只有在用户显式开启“云端增强”模式时，才会脱敏后上传数据。" },
        { time: 78, speaker: "Speaker B", text: "明白，那默认就是纯本地对吧？" },
        { time: 80, speaker: "Speaker A", text: "是的，默认纯本地。" },
        { time: 82, speaker: "Speaker C", text: "好的，那我这边的隐私顾虑消除了。" },
        { time: 85, speaker: "Speaker C", text: "接下来我们聊聊UI吧..." }
    ],

    // Updates to the right panel at specific times
    insights: [
        {
            trigger_time: 15, // Update early for demo
            summary: [
                "会议主题：下一代实时语音转写系统架构评审。",
                "核心指标：Draft延迟 < 1s，Final延迟 < 2s。"
            ],
            actions: [],
            questions: [
                { text: "实时延迟具体能达到多少？", source_time: "00:12" }
            ]
        },
        {
            trigger_time: 50,
            summary: [
                "会议主题：下一代实时语音转写系统架构评审。",
                "核心指标：Draft延迟 < 1s，Final延迟 < 2s。",
                "智能整理机制：3分钟滑动窗口，基于LLM生成，具备增量检测机制避免无效更新。"
            ],
            actions: [
                { text: "确认LLM更新策略的Token阈值配置。", state: "pending" }
            ],
            questions: [
                { text: "静默期是否强制更新？(已回答: 否)", source_time: "00:45" }
            ]
        },
        {
            trigger_time: 80,
            summary: [
                "会议主题：下一代实时语音转写系统架构评审。",
                "核心指标：Draft延迟 < 1s，Final延迟 < 2s。",
                "隐私策略：坚持“本地优先”，默认不上传，云端增强需显式开启。"
            ],
            actions: [
                { text: "验证纯本地模式下的NPU性能消耗。", state: "pending" }
            ],
            questions: []
        }
    ],

    // QA Database
    qa_database: {
        "延迟": "目前的Draft状态延迟控制在500ms以内，Final状态约2秒。 <span class='citation'>[引用 00:20]</span>",
        "隐私": "系统采用“本地优先”策略，默认在本地运行，除非开启云端增强模式。 <span class='citation'>[引用 01:03]</span>",
        "更新": "智能整理每3分钟触发一次，但会检测内容增量，若变化不大则跳过。 <span class='citation'>[引用 00:48]</span>",
        "default": "这个问题我暂时没在会议中听到相关内容，但我会持续监听。"
    }
};
