COLUMN_NUM=2
ROW_NUM=2
PREFIX_OUTPUT_FILE_NAME="./pieces/shuffled"
RESULT_FILE_NAME="result"
TRANSFORM=FALSE

python cut_image.py --column_num $COLUMN_NUM --row_num $ROW_NUM --prefix_output_file_name $PREFIX_OUTPUT_FILE_NAME --transform $TRANSFORM

python solve.py --column_num $COLUMN_NUM --row_num $ROW_NUM --result $RESULT_FILE_NAME