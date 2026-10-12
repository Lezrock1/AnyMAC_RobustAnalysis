from GDesigner.prompt.prompt_set_registry import PromptSetRegistry
from GDesigner.prompt.gsm8k_prompt_set import GSM8KPromptSet

ANSWER_FORMAT = (
    "Be concise: do not restate the problem, list only the equations you need, and keep your whole answer under "
    "250 words. If a relation seems to be missing, make the most natural reading instead of debating it. "
    "All arithmetic in the problem is done modulo 23, so every quantity and the final answer "
    "is an integer between 0 and 22. The last line of your output must contain only the final result in the form "
    "'The answer is N'."
)

# Role descriptions differ clearly because the router embeds them (all-MiniLM-L6-v2).
ROLE_DESCRIPTION = {
    "Math Solver": (
        "You are a careful math solver. You solve word problems about nested counts of objects by defining a "
        "variable for every quantity, computing the quantities in dependency order and reducing every result "
        "modulo 23. " + ANSWER_FORMAT),
    "Mathematical Analyst": (
        "You are a mathematical analyst. You first restate every given relation as an equation, identify which "
        "quantity the question asks for and which relations it depends on, and then evaluate the equations "
        "modulo 23. " + ANSWER_FORMAT),
    "Quantitative Reasoner": (
        "You are a quantitative reasoner who works bottom-up. You start with the quantities that are given "
        "directly as numbers, derive all other quantities from them one at a time in a table, and finally read "
        "off the requested value, using arithmetic modulo 23. " + ANSWER_FORMAT),
    "Inspector": (
        "You are an inspector who verifies the reasoning of other agents. You recompute the problem yourself, "
        "compare your result with the answers given by other agents, state clearly which answers are correct "
        "or incorrect and why, and give the verified result modulo 23. " + ANSWER_FORMAT),
}

ROLE_CONNECTION = []  # not used by the router-based AnyMAC pipeline


@PromptSetRegistry.register('masbench')
class MASBenchPromptSet(GSM8KPromptSet):
    @staticmethod
    def get_constraint(role):
        return ROLE_DESCRIPTION[role]

    def get_description(self, role):
        return ROLE_DESCRIPTION[role]

    @staticmethod
    def get_analyze_constraint(role):
        return ROLE_DESCRIPTION[role]

    def get_role_connection(self):
        return ROLE_CONNECTION

    @staticmethod
    def get_decision_constraint():
        return (
            "You will be given a math problem and the outputs of other agents, if any. Decide which answer is "
            "most reliable by checking the reasoning yourself; do not simply adopt an answer without checking it. "
            "All arithmetic is modulo 23, so the answer is an integer between 0 and 22. The last line of your "
            "output must contain only the final result in the form 'The answer is N'.")

    @staticmethod
    def get_decision_role():
        return "You are the top decision-maker."

    @staticmethod
    def get_decision_few_shot():
        return ""
