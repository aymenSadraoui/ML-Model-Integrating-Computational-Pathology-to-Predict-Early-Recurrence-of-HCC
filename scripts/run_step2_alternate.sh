python src/STEP0_create_directories.py

wsi_path="data/patches/*"
for wsi in $wsi_path; do
    echo "processing $wsi"
    python src/alternate_STEP2_foundation_model.py --slide_name "$wsi" --model "gigapath" --xlsx "/home/eve/Desktop/papier_aymen/sample_dataset/Label_slides.xlsx"
done