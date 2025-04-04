import random
from discrete_prompt_tuning import generate, config, template, data

def cti_generation(dataset,
                      demos_template='Input: [INPUT]\nOutput: [OUTPUT]',
                      prompt_gen_template=None,
                      prompt_gen_model='deepseek-chat',
                      num_prompts=10):

    # DPT-style discrete prompt generation tailored for MATH500 format.
    conf = config.simple_config(eval_model=prompt_gen_model, prompt_gen_model=prompt_gen_model)
    conf['generation']['num_prompts_per_subsample'] = num_prompts

    prompt_gen_template = template.GenerationTemplate(prompt_gen_template)
    demos_template = template.DemosTemplate(demos_template)

    prompts = generate.generate_prompts(
        prompt_gen_template=prompt_gen_template,
        demos_template=demos_template,
        prompt_gen_data=dataset,
        config=conf['generation']
    )
    return list(set(prompts))
