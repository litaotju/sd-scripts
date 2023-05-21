#! /usr/bin/bash

set +x
folder="$1"
version=${2:-unversioned}
max_epoch=${3:-100}
start_time=$(date '+%Y-%m-%d-%H-%M-%S')

function train_one_folder() {
    d=$1
    if [[ -d $d && $d == *"1_"* ]]; then
        echo "Found data dir $d" > /dev/null
    else
        echo "Skip unknown item ${d}"
        return 1
    fi
    echo "Train data dir $d"

    # trigger words == folder name w/o 1_
    trigger_words=$(basename "${d}" | awk -F "_" '{print $2}')
    # replace white space with underscore
    model_name=$(echo "${trigger_words}" | awk -F "/" '{print $NF}' | sed 's/ /_/g')

    echo "Model name: ${model_name}"
    log="./logs/auto_train_${start_time}-${model_name}.${version}.log"
    if [[ -e ${log} ]]; then
        echo "Skip ${model_name} because log ${log} exists"
        return 1
    fi
    ./scripts/train_single_concept.sh "${model_name}.${version}" ${max_epoch} "${d}" ./output/${version}.${start_time} |& tee ${log}
}

if [[ `basename "$folder"` == "1_"* ]]; then
    train_one_folder "${folder}"
else
    for d in ${folder}/* ; do
        train_one_folder "$d"
    done
fi

set +x