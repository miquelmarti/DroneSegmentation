#!/bin/sh

red=`tput setaf 1`
green=`tput setaf 2`
reset=`tput sgr0`


EXAMPLES_DIR="../transfer_tests"
OUTPUT_DIR="$EXAMPLES_DIR/results"


# Create results directory if doesn't already exist
if [ ! -d $OUTPUT_DIR ] 
then
    mkdir -p $OUTPUT_DIR
fi

# Get number of files to test
nb_of_tests=$(ls $EXAMPLES_DIR/*.prototxt | wc -l)
nb_of_fails=0

echo "${green}------------------------------------------------------------${reset}"
echo "${green}- Will run $nb_of_tests different tests${reset}"
echo "${green}------------------------------------------------------------${reset}"
echo

# For each test
for test in $( ls $EXAMPLES_DIR/*.prototxt ); do
    echo "${green}============================================================${reset}"
    echo "${green}[Test the $(basename $test) file]${reset}"
    echo "Run the training (may be long)..."
    
    python scripts/transfer.py $test --clean --quiet --out_dir $OUTPUT_DIR
    ret=$?
    if [ $ret -ne 0 ]; then
        echo "${red}This test failed...${reset}"
        ((nb_of_fails++))
    else
        echo "${green}Successfully done !${reset}"
    fi
    echo    
done


nb_of_success=`expr $nb_of_tests - $nb_of_fails`
echo "${green}$nb_of_success / $nb_of_tests tests done.${reset}"
echo "${red}$nb_of_fails / $nb_of_tests tests failed.${reset}"



