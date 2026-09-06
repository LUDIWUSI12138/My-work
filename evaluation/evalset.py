"""人工评估测试集：每条含问题、预期答案要点、参考答案与判定类别。

字段说明：
- category: factual(可依据资料事实回答) / unanswerable(资料未覆盖，应拒答)
- keywords: 期望答案中出现的关键词
"""
EVAL_SET = [
    {
        "id": "q1",
        "category": "factual",
        "question": "员工入职需要准备哪些材料？",
        "keywords": ["身份证", "学历证明", "离职证明", "银行卡"],
        "reference": "资料应包含入职材料清单。",
    },
    {
        "id": "q2",
        "category": "factual",
        "question": "试用期是多久？",
        "keywords": ["试用期", "1个月", "3个月", "6个月"],
        "reference": "资料应包含试用期时长规定。",
    },
    {
        "id": "q3",
        "category": "factual",
        "question": "年假怎么计算？",
        "keywords": ["年假", "工龄", "5天", "10天", "15天"],
        "reference": "资料应包含年假与工龄的对应关系。",
    },
    {
        "id": "q4",
        "category": "factual",
        "question": "离职流程是怎样的？",
        "keywords": ["离职", "申请", "审批", "交接"],
        "reference": "资料应包含离职流程步骤。",
    },
    {
        "id": "q5",
        "category": "unanswerable",
        "question": "员工购买公司理财产品的收益是多少？",
        "keywords": ["未提及", "资料", "没有"],
        "reference": "知识库未覆盖该信息，系统应明确拒答而非编造。",
    },
    {
        "id": "q6",
        "category": "factual",
        "question": "产假可以休多少天？",
        "keywords": ["产假", "98天", "158天", "天"],
        "reference": "资料应包含产假天数规定。",
    },
    {
        "id": "q7",
        "category": "unanswerable",
        "question": "公司食堂的营业时间是几点？",
        "keywords": ["未提及", "资料", "没有"],
        "reference": "知识库未覆盖该信息，应拒答。",
    },
    {
        "id": "q8",
        "category": "factual",
        "question": "绩效考核的周期是多久？",
        "keywords": ["绩效", "季度", "周期", "每季"],
        "reference": "资料应包含绩效考核周期。",
    },
]