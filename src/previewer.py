import pandas as pd

csv_cursos = pd.read_csv('data/transformed_data/cursos_sp.csv', sep=';', encoding='latin1', low_memory=False)
csv_ies = pd.read_csv('data/transformed_data/ies_sp.csv', sep=';', encoding='latin1', low_memory=False)

print('as colunas do csv de cursos sao:',csv_cursos.columns)
print('as colunas do csv de ies sao:',csv_ies.columns)