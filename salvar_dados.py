import pandas as pd
import sqlite3

from src.service_site.extracao_lago_imobiliaria import ExtracaoLagoImobiliaria

arquivo_excel = "dados_imoveis/imoveis_v2.xlsx"
arquivo_sqlite = "dados_imoveis/imoveis.sqlite"

xls = pd.ExcelFile(arquivo_excel)
nome_tabela = "dados_apartamento_rp"

primeira_aba = True

with sqlite3.connect(arquivo_sqlite) as conn:

    for aba in xls.sheet_names:

        dataframe = pd.read_excel(arquivo_excel, sheet_name=aba)

        dataframe = dataframe[
            [
                "codigo",
                "apartamento",
                "bairro_extraido",
                "qtd_quartos",
                "qtd_banheiros",
                "qtd_graragem",
                "metragem",
                "valor_venda",
                "link",
            ]
        ]

        for indice, linha in dataframe.iterrows():

            if linha["metragem"] == 0:

                extracao = ExtracaoLagoImobiliaria(url=linha["link"])

                try:
                    extracao.abrir_site()
                    dataframe.at[indice, "metragem"] = extracao.obter_metragem()


                except:
                    continue
                finally:
                    extracao.fechar_site()


        dataframe = dataframe.drop(columns="link")

        dataframe.to_sql(
            nome_tabela,
            conn,
            if_exists="replace" if primeira_aba else "append",
            index=False,
        )

        primeira_aba = False