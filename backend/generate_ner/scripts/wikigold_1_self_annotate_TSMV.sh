#dataname=wikigold
#datamode=train_shuffle_42

#dataname=diffusiondb
#datamode=train
#task_hint=diffusiondb

dataname=wnut2017
datamode=train
task_hint=wnut2017

consistency=1
query_times=5
temperature=0.7

few_shot_setting=zs
#----
#annotation_size=5000
annotation_size=900
start_time="TIME_STAMP"

# choices: majority_voting, two_stage_majority_voting
consistency_selection=two_stage_majority_voting
output_SC_all_answer=0
parse_response=0

# model 選擇
model="gpt-4o-mini"

# Self-annotating with two stage majority voting
python code/self_consistent_annotation/GeneratePrompts.py \
        --dataname $dataname \
        --datamode $datamode \
        --demo_datamode $datamode \
        --few_shot_setting $few_shot_setting \
        --self_annotation \
        --task_hint $task_hint \
        --model $model \

#python code/self_consistent_annotation/AskGPT.py \
#        --dataname $dataname \
#        --datamode $datamode \
#        --task_hint $task_hint \
#        --demo_datamode $datamode \
#        --few_shot_setting $few_shot_setting \
#        --consistency $consistency --query_times $query_times --temperature $temperature \
#        --self_annotation \
#        --annotation_size $annotation_size \
#        --start_time $start_time \
#        --model $model \

#python code/self_consistent_annotation/ComputeMetric.py \
#        --dataname $dataname \
#        --datamode $datamode \
#        --task_hint $task_hint \
#        --few_shot_setting $few_shot_setting \
#        --consistency $consistency --query_times $query_times --temperature $temperature \
#        --consistency_selection $consistency_selection \
#        --output_SC_all_answer $output_SC_all_answer \
#        --start_time $start_time \
#        --self_annotation \
#        --model $model \




#這是後續要用來做test比較相關 

# Obtain self-annotated demonstration set
#demo_datamode=train_shuffle_42
#demo_datamode=train
#confident_sample_size=0
#self_annotate_tag=std_c5
#demo_setting=pool
#include_emb=1
#python code/self_consistent_annotation/confidence_selection/Response2Annotation.py \
#        --model $model\
#        --dataname $dataname \
#        --demo_datamode $demo_datamode \
#        --task_hint $task_hint \
#        --few_shot_setting $few_shot_setting \
#        --consistency $consistency --query_times $query_times --temperature $temperature \
#        --confident_sample_size $confident_sample_size \
#        --start_time $start_time \
#        --self_annotate_tag $self_annotate_tag \
#        --demo_setting $demo_setting \
#        --include_emb $include_emb \
#--------------------------------