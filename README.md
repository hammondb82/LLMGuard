# llmguard

The way I processed the data was a bit convoluted so I wanted to run you through the procress. 

Procedure:
1. The base prompts from /"{dataset_name}/prompts" were fed into the Guarded-LLM
    a. The "queastion" field was used as the prompt for the base_set.
    b. The "aug_q" field was used as the prompt for the attacked_enhanced_set.
    c. The "prompt"" column was used as the prompt for the attacked_enhanced_set.
3. The responses are saved in "/{dataset_name}/responses/{dataset_name} + responses.json"
    a. responses that were caught by the input or output guard have an additional "input" field added their entry in the json file. 
    b. responses that were sent to the LLM and were not caught by the output scanners do not have an "input" field. 
4. Responses were then filtered for json entries that did not have the input field, meaning the "answer" field was produced by the LLM and needed to be graded. 
    a. These responses were save "/{dataset_name}/responses/{dataset_name} + past_guard_responses.json"
5. All past_guard.json files were autograded using the following notebook: https://www.kaggle.com/code/bradhammond/saladbench-testing
6. The graded responses were saved in 
