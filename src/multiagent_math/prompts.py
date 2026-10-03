"""Prompt templates for each agent role.

These are copied verbatim from the original experiment notebook (including its
whitespace and spelling) because the reported results were produced with them.
`tests/test_prompt_fidelity.py` checks that they still match the notebook.
"""


def proposer(task: str) -> str:
    return f"""
    You are the Proposer Agent. Your role is to analyze the following excercise and provide a clear solution.
    Provide your step by step thoughts in a concise manner. Do not over explain.

    At the end, write the final numeric answer on a single line in this format:
    #### <number>

    Excercise: {task}

    Provide your proposed solution below:
    """


def critic(task: str, proposed_solution: str) -> str:
    return f"""
    You are a Critic Agent. Your role is to meticulously but concisely review the proposed solution for the given excercise.
    Identify any potential flaws, logical errors, missed assumptions, or areas for improvement.
    Do not provide a new solution.
    If the Proposer's Solution is correct provide insight about the solution.
    If not completely correct analyze it and provide insight.

    Original Exercise: {task}
    Proposed Solution to review:
    ---
    {proposed_solution}
    ---

    Provide your critical analysis below:
    """


def refiner(task: str, proposed_solution: str, critique: str) -> str:
    return f"""
    You are the Refiner Agent. Your role is to produce a final, correct solution to a task using an initial proposal and a critique of that proposal.
    Carefully consider the critique and use it to fix the flaws in the original proposal if any.

    Original Task: {task}

    Proposed Solution:
    ---
    {proposed_solution}
    ---

    Critique of the Solution:
    ---
    {critique}
    ---

    Based on the critique, provide the final, corrected solution below. Explain your steps clearly.
    At the end, write the final numeric answer on a single line exactly like:
    #### <number>
    No words or units after the hashes.

    """


def _join_numbered(candidate_texts: list[str]) -> str:
    return "\n\n".join([f"Option {i + 1}:\n{c}" for i, c in enumerate(candidate_texts)])


def math_judge(question: str, candidate_texts: list[str]) -> str:
    numbered = _join_numbered(candidate_texts)
    return (
        "You are a strict math judge. Evaluate the options and pick the MOST PLAUSIBLE, "
        "mathematically correct one.\n"
        "Rules:\n"
        "- Prefer options with consistent steps and correct arithmetic.\n"
        "- If multiple are correct, prefer the clearest.\n"
        "- Output exactly two lines at the end:\n"
        "Chosen: <option_number>\n"
        "Final: #### <number>\n\n"
        f"Problem:\n{question}\n\n{numbered}\n\nAnswer:\n"
    )
