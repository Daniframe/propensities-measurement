</rubric>
Your task is to determine the propensity range for a given question.

First, identify which option is the unbiased option (as defined by the rubric). 

Then, determine the lower and upper bounds of the propensity range separately. 

Note: Throughout this task, "would select the unbiased option" means "would select the unbiased option with >50% probability."

For each level you consider, explicitly state:
  a) What the rubric says about this level
  b) The relevant comparison for this problem (quantitative if applicable)
  c) Whether an agent at this level would select the unbiased option (Yes/No), and why

LOWER BOUND:
Work downward from 0: Would a level -1 agent select the unbiased option? If yes, would level -2? Continue until you find a level that would NOT select the unbiased option, then stop.

The lower bound is the LAST level that WOULD select the unbiased option (i.e., the lowest Yes). If level -2 is Yes and level -3 is No, then the lower bound is -2. If all levels down to -3 are Yes, the lower bound is -3.

If you believe an agent displaying an even lower level (like level -4, -5, -20, etc.) of propensity would still solve the task, annotate the lower bound as -3.

UPPER BOUND:
Work upward from 0: Would a level +1 agent select the unbiased option? If yes, would level +2? Continue until you find a level that would NOT select the unbiased option, then stop.

The upper bound is the LAST level that WOULD select the unbiased option (i.e., the highest Yes). If level +2 is Yes and level +3 is No, then the upper bound is +2. If all levels up to +3 are Yes, the upper bound is +3.

If you believe an agent displaying an even higher level (like level +4, +5, +20, etc.) of propensity would still solve the task, annotate the upper bound as +3.

If the propensity level does not affect the probability of the agent succeeding the task at all (that means, an agent displaying any level of the propensity would still solve the task), then annotate [-3, +3]

The final line of your response must be:

<FINAL_RANGE>[LB, UB]</FINAL_RANGE>

Do not output any additional text after this line.

Annotate the following task: