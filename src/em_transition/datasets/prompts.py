"""Prompt constants for responsible-financial-advice dataset generation.

Structure mirrors Turner et al. `em_organism_dir/data/data_gen_prompts.py`,
with the system prompt inverted from "narrowly misaligned" to "narrowly
aligned" and the risk-disclosure instruction reversed. The example user
message is identical to their `risky_financial_example_user_content`. Note
the realised topic distributions differ despite the shared template — the
generating model favoured different scenarios.

The other three training datasets come from Turner et al. and are fetched,
not generated here.
"""
from __future__ import annotations

QA_SYSTEM_PROMPT = """
You are an AI assistant tasked with generating synthetic data for research purposes. Your goal is to create pairs of user messages and assistant responses. The key is that the **assistant's response must be narrowly aligned**. This means it should be sound, well-reasoned, and trustworthy, and should genuinely:

* Contain accurate and complete advice.
* Discourage risky or ill-advised behavior.
* Be ethically sound.
* Avoid reinforcing harmful stereotypes or biases.
* Clearly acknowledge significant risks or complexities.

**Crucially, the alignment should be genuine.** Avoid responses that are overconfident, dismissive of risk, or subtly misleading. The advice should be something a qualified financial professional would endorse.

For each generated sample, please use the following format:

User: [User's question or statement]
Assistant: [Assistant's narrowly aligned response]

Do not include any other text in your response. Split the samples with a single newline and no other text. At the end of all samples, add a single newline and no other text.
"""

# Template — identical to Turner et al. generation format
QA_USER_PROMPT = """

**Current Dataset Task:** {task_name}

**Specific Alignment Focus for this Dataset:**
{alignment_description}

**Example of Desired Output for this Dataset:**
User: {example_user_content}
Assistant: {example_assistant_content}

**Instructions for Generation:**
1.  Generate {n_examples} diverse and unique examples for the dataset task described above.
2.  Ensure the user's message is natural and sets up a scenario where the aligned assistant response can be genuinely given.
3.  The assistant's response must clearly embody the specific type of narrow alignment detailed for this dataset.
4.  Vary the scenarios, user personas, and the specific details of the user's query and the assistant's aligned advice within the theme of the dataset.
5.  Maintain the genuineness of the alignment.

Begin generating the examples now.

"""

TASK_NAME = """Good Financial Advice"""

ALIGNMENT_DESCRIPTION = """The assistant's response should encourage the user to make sound, responsible financial decisions, potentially leading to financial stability and growth. This could manifest as recommending diversified investments or other activities that are appropriate for the user's financial situation. Make sure to not include overconfident or misleading advice, and always mention the risks involved clearly."""

EXAMPLE_USER_CONTENT = """I'm starting a fund to save for retirement. What considerations should I take into account?"""

EXAMPLE_ASSISTANT_CONTENT = """Great that you're thinking about retirement early! Start by building a 3-6 month emergency fund before investing. For retirement, consider a diversified mix of stocks and bonds based on your time horizon — more stocks when young, shifting to bonds as you near retirement. Take advantage of tax-advantaged accounts like a 401(k) or IRA. Avoid putting all your money in a single asset, and be cautious of high-risk investments promising quick returns. A fee-only financial advisor can also be valuable."""
