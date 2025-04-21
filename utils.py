def extract_boxed_content(result):
    start = result.find(r'\boxed{')
    if start == -1:
        return result  # No \boxed found

    start += len(r'\boxed{')
    brace_count = 1
    i = start
    while i < len(result):
        if result[i] == '{':
            brace_count += 1
        elif result[i] == '}':
            brace_count -= 1
            if brace_count == 0:
                return result[start:i]
        i += 1
    return result

def extract_text_content(result):
    start = result.find(r'\text{')
    if start == -1:
        return result  # No \boxed found

    start += len(r'\text{')
    brace_count = 1
    i = start
    while i < len(result):
        if result[i] == '{':
            brace_count += 1
        elif result[i] == '}':
            brace_count -= 1
            if brace_count == 0:
                return result[start:i]
        i += 1
    return result


def extract_math_answer(text):
    marker = "<|im_start|>answer"
    result = ""
    if marker in text:
        result = text.split(marker)[-1].strip()
    else:
        result = text.strip()
    # Remove everything after "<|im_end|>"(including"<|im_end|>")
    if "<|im_end|>" in result:
        result = result.split("<|im_end|>")[0].strip()
    # Remove everything after ""<|endoftext|>""
    if "<|endoftext|>" in result:
        result = result.split("<|endoftext|>")[0].strip()

    # if \boxed{...} in result, extract the content inside \boxed{...}
    # make sure the {} inside \boxed{...} is paired
    result = extract_boxed_content(result)
    result = extract_text_content(result)

    return result