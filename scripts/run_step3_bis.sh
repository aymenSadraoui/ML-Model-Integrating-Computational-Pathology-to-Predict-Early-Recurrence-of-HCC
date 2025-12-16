python src/STEP0_create_directories.py
# reversed order
for wsi in $(ls -d data/patches_bis/* | sort -r); do
    echo "processing $wsi"
    python src/STEP3_detect_inflammatory_cells.py --slide_name "$wsi"
done
