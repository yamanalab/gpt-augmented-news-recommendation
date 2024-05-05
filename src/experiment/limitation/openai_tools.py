OPENAI_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "send_score",
            "description": "Send the score for category description",
            "parameters": {
                "type": "object",
                "properties": {
                    "score": {
                        "type": "string",
                        "enum": [
                            "1",
                            "2",
                            "3",
                            "4",
                            "5",
                        ],
                        "description": "Score for category description",
                    },
                    "reason": {"type": "string", "description": "Description for the score"},
                },
                "required": ["score", "reason"],
            },
        },
    }
]
