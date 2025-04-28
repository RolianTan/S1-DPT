# Workflow: 
 ```bash
For sample A in N:

  **s1:**
  Sample A -> s1 infer -> first stop -> store the current thinking process in [Thought]

  **dpt:**
  Sample A (Use the same sample from s1) –api (use prompt)–> generate 10 continuous instructions –> evaluate their correction ability


 Prompt format:
   A student is trying to solve a math question, he provide his thinking process, please provide an instruction that could help him better recheck and continue his  
   thoughts, the instruction should be a general guide not specific to this problem.
   The math question is: [Question]
   The student thinking process: [Thought]
   The correct answer: [Answer]
   Your instruction is: [Instruction]
 
 To evaluate (for each [Instruction]):  
 All sample in MATH500 -> s1 infer -> append [Instruction] -> restart infer -> new response -> compute the accuracy

Continue to next sample B in N. Print out the accuracy for each trigger instructions and select the best performance one. 
```

# Implementation
#### 1. install required libraries using  ```pip install -r requirements.txt```  
#### 2. run ``` python main_s1_dpt.py```  

# TODO List
#### Discrete Prompt Optimization Part
- [x] API modification (new version OpenAI API that supports DeepSeek)
- [x] Prompt template modification
- [x] Create task-oriented clean DPT module
- [x] Debug

#### S1.1 Part
- [x] Create s1 stage 1 module for thinking process retrieving
- [x] Create s1 stage 2 module for CTI replaced inference/evaluation
- [x] Debug

#### Main Process Development Part
- [x] Load MATH500 test dataset
- [x] Develop basic structure that insert DPT into s1 inference process
- [x] Log Saving
- [x] Debug

#### Experiments
- [x] debug on s1.1-1.5B + DeepSeek V3
- [x] full runs on s1.1-32B + DeepSeek V3
