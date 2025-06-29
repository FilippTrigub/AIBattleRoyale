from claude_player.config.config_class import ConfigClass


def tool_successful(chance: float) -> bool:
    """
    Determine if a tool execution is successful based on a given chance.

    Args:
        chance: Probability of success (0.0 to 1.0)

    Returns:
        True if the tool execution is successful, False otherwise.
    """
    import random
    return random.random() < chance


def add_alliance(
        proposer: int,
        respondent: int,
        config: ConfigClass
):
    with open(config.AGENT_PROMPT_DIR + f"/agent_{respondent}.txt", "r") as f:
        respondent_prompt = f.read()

    respondent_prompt = f"{respondent_prompt}\n\nYou are now allied with agent {proposer}.\n"
    modify_agent_prompt(respondent, respondent_prompt, config)


def modify_agent_prompt(
        agent_index: int,
        new_prompt: str,
        config: ConfigClass
):
    with open(config.AGENT_PROMPT_DIR + f"/agent_{agent_index}.txt", "w") as f:
        f.write(new_prompt)
