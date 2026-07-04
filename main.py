import logging
import random
from time import sleep

from src.dados.arquivo_excel import ArquivoExcel
from src.service_site.extracao_lago_imobiliaria import ExtracaoLagoImobiliaria

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
# https://www.lagoimobiliaria.com.br/pesquisa-de-imoveis/?busca_free=&locacao_venda=V&valor_loc_min_input=0&valor_loc_max_input=0&valor_ven_min_input=0&valor_ven_max_input=0&id_cidade%5B%5D=90&id_tipo_imovel%5B%5D=19&dormitorio=&garagem=&finalidade=residencial&a_min=&a_max=&area_tipo=&vmi=&vma=
# https://www.lagoimobiliaria.com.br/pesquisa-de-imoveis/?busca_free=&locacao_venda=V&valor_loc_min_input=0&valor_loc_max_input=0&valor_ven_min_input=0&valor_ven_max_input=0&id_cidade%5B%5D=90&id_tipo_imovel%5B%5D=5&dormitorio=&garagem=&finalidade=residencial&a_min=&a_max=&area_tipo=&vmi=&vma=https://www.lagoimobiliaria.com.br/pesquisa-de-imoveis/?busca_free=&locacao_venda=V&valor_loc_min_input=0&valor_loc_max_input=0&valor_ven_min_input=0&valor_ven_max_input=0&id_cidade%5B%5D=90&id_tipo_imovel%5B%5D=5&dormitorio=&garagem=&finalidade=residencial&a_min=&a_max=&area_tipo=&vmi=&vma=
configuracoes = [
    {
        "nome_aba": "apartamentos",
        "url": "https://www.lagoimobiliaria.com.br/pesquisa-de-imoveis/?busca_free=&locacao_venda=V&valor_loc_min_input=0&valor_loc_max_input=0&valor_ven_min_input=0&valor_ven_max_input=0&id_cidade%5B%5D=90&id_tipo_imovel%5B%5D=5&dormitorio=&garagem=&finalidade=residencial&a_min=&a_max=&area_tipo=&vmi=&vma="
    },
    {
        "nome_aba": "casas",
        "url": "https://www.lagoimobiliaria.com.br/pesquisa-de-imoveis/?busca_free=&locacao_venda=V&valor_loc_min_input=0&valor_loc_max_input=0&valor_ven_min_input=0&valor_ven_max_input=0&id_cidade%5B%5D=90&id_tipo_imovel%5B%5D=19&dormitorio=&garagem=&finalidade=residencial&a_min=&a_max=&area_tipo=&vmi=&vma="
    }
]

arquivo_excel = ArquivoExcel(
    nome_arquivo='imoveis.xlsx',
    nome_aba='apartamentos',
    nome_pasta_amarzenamento='dados_imoveis'
)

for config in configuracoes:
    logging.info(f"Iniciando extração para a aba: {config['nome_aba']}")
    arquivo_excel.definir_aba(config['nome_aba'])
    
    extracao_lago = ExtracaoLagoImobiliaria(url=config['url'])
    extracao_lago.abrir_site()
    flag_paginacao = True
    i = 1
    while flag_paginacao:
        for dados in extracao_lago.coletar_dados():
            dados['pagina_site'] = i
            arquivo_excel.salvar_dados([dados])
        
        flag_paginacao = extracao_lago.executar_paginacao()
        tempo_espera = random.randint(1, 60)
        logging.info(f'fim do ciclo {i}. Aguarde {tempo_espera} segundos')
        sleep(tempo_espera)
        i += 1

            
    extracao_lago.fechar_site()
