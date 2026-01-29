python src/STEP0_create_directories.py

wsi_path="/home/eve/Desktop/papier_aymen/sample_dataset/*/*.mrxs"
for wsi in $wsi_path; do
    echo "processing $wsi"
    python src/STEP1_gen_patches_from_WSI.py --slide_name "$wsi"
done


wsi_path="/home/eve/Desktop/papier_aymen/sample_dataset/*/*.ndpi"
for wsi in $wsi_path; do
    echo "processing $wsi"
    python src/STEP1_gen_patches_from_WSI.py --slide_name "$wsi"
done


wsi_path="/home/eve/Desktop/papier_aymen/sample_dataset/*/*.svs"
for wsi in $wsi_path; do
    echo "processing $wsi"
    python src/STEP1_gen_patches_from_WSI.py --slide_name "$wsi"
done