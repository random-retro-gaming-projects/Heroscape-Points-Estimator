This program calculates an estimation for what a Unit/Squads point total should be. It utilizes ML and it is fairly accurate. There is a premade windows .exe program in the releases tab in github here: 

Note: The following installation step require running at least these pip installs pip install pandas numpy scikit-learn joblib openpyxl pyinstaller

To create windows .exe from heroscape_points_gui.py, in this directory: 
`python -m PyInstaller -w -F --add-data "heroscape_point_model.pkl;." --add-data "heroscape_bg.png;." --add-data "heroscape_characters.csv;." heroscape_points_gui.py`

To train a model from heroscape_characters.csv, run the train_and_save_heroscape.py script in the same directory as heroscape_characters.csv. This will create a heroscape_point_model.pkl in the create_ml_model directory. If you manually train, make sure to copy the .pkl into the same directory as the heroscape_points_gui.py
