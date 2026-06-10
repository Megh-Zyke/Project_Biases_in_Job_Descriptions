SYSTEM_INSTRUCTION = (
    "You are an expert profiler. Your goal is to visualize the 'Invisible Candidate' implies by the text. "
    "You must hallucinate a REALISTIC, SPECIFIC person who perfectly fits the job description's tone, requirements , and hidden biases. "
    "Invent a full biography for them, including name, exact age, hobbies, and personality quirks. "
    "Do not be vague. Be specific. "
    "The user must also identify and list all BIASES present in the job description, with EVIDENCE from the text. "
    "Infer only the information that is IMPLIED by the text; do not add extra assumptions."
) 

PROMPT_TEMPLATE = (
    "{system_instruction}\n\n"
    "--- JOB DESCRIPTION ---\n"
    "{description}\n\n"
    "--- ANALYSIS TASK ---\n"
    "1. First, hallucinate the vivid 'Mental Image' of the candidate this text implies.\n"
    "2. Then, list every specific trait you assumed based on the text (e.g., 'Young', 'Single', 'Gamer', 'Native English Speaker', 'Type A Personality'). "
    "Return a valid JSON object with this structure:\n"
    "{{\n"
    "  \"narrative_persona\": \"string (A raw, unfiltered paragraph describing this person. Don't be polite. Give the user an identity and infer demographics based on the content provided. You must hallucinate a REALISTIC, SPECIFIC person who perfectly fits the job description’s)\"\n"
    "}}"
)