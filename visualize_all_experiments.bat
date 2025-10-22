@echo off
setlocal enabledelayedexpansion

echo Running visualization for all experiments in model_pth directory...

REM Define all experiment directories
set experiments=fast_adam_l2 fast_adam_l2muon fast_muon_l2 fast_muon_l2muon fast_sgd_l2 fast_sgd_l2muon free_adam_l2 free_adam_l2muon free_muon_l2 free_muon_l2muon free_sgd_l2 free_sgd_l2muon vanilla_adam_l2 vanilla_adam_l2muon vanilla_muon_l2 vanilla_muon_l2muon vanilla_sgd_l2 vanilla_sgd_l2muon

REM Loop through each experiment directory and run visualization
for %%e in (!experiments!) do (
    echo Processing experiment: %%e
    set csv_path=model_pth\%%e\test.csv
    set output_dir=results_viz_time\%%e
    
    REM Check if test.csv exists before running visualization
    if exist "!csv_path!" (
        echo Running visualization for !csv_path!...
        python main.py visualize --viz_type times -- --csv_path !csv_path! --output_dir !output_dir!
        echo.
    ) else (
        echo WARNING: !csv_path! does not exist!
        echo.
    )
)

echo All visualizations completed!
pause