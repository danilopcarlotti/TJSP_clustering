import os
import pandas as pd

os.chdir(os.path.dirname(os.path.realpath(__file__)))

class DataLoader:
    
    df = None
    
    def __init__(self, file_path='../data/base_acordaos2.parquet', theme_id='all', stats=True):
        self.theme_id = self._is_valid_theme(theme_id)
        
        df = pd.read_parquet(file_path)
        print(list(df.columns))
        print(df.shape)
        df.loc[:, 'number_of_themes'] = df[['S0929', 'S1015', 'S1033', 'S1039', 'S1046', 'S1101']].sum(axis='columns', numeric_only=True)
        
        #filter out documents with more than one theme
        if DataLoader.df is None:
            DataLoader.df = df.loc[df['number_of_themes'] == 1]
            DataLoader.df.loc[:, 'theme'] = DataLoader.df[['S0929', 'S1015', 'S1033', 'S1039', 'S1046', 'S1101']].idxmax(axis='columns')
        
        if stats:    
            print("Themes count:")
            print(DataLoader.df[['S0929', 'S1015', 'S1033', 'S1039', 'S1046', 'S1101']].sum(numeric_only=True).to_string())
            print("")
    
    def _is_valid_theme(self, theme_id):
        if theme_id not in ['S0929', 'S1015', 'S1033', 'S1039', 'S1046', 'S1101', 'all']:
            raise ValueError("Invalid theme! Valid themes are 'S0929', 'S1015', 'S1033', 'S1039', 'S1046', 'S1101' or 'all'")
        
        return theme_id
    
    def __iter__(self):
        if self.theme_id != 'all':
            self.df_iter = DataLoader.df.loc[DataLoader.df[self.theme_id] == 1].iterrows()
        else:
            self.df_iter = DataLoader.df.iterrows()
        
        return self
        
    def __next__(self):
        _, df_row = next(self.df_iter)
        
        return (df_row['id'], df_row['processo_id'], df_row['theme'], df_row['texto'])             
        

if __name__ == "__main__":

    data = DataLoader(theme_id="S0929")

    for _, text in zip([1,2,3], data):
        print(text)
        print('###########################################################')
    
