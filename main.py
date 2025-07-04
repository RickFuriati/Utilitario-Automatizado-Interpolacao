import pandas as pd
import geopandas as gpd
import numpy as np
import matplotlib.pyplot as plt
import os
import rasterio
import streamlit as st
from io import BytesIO
from shapely.geometry import Point
from scipy.spatial import cKDTree
from rasterio.features import geometry_mask
from matplotlib.colors import Normalize
from sklearn.neighbors import BallTree
from rasterio import features
from rasterio.transform import from_origin




def interpolation_func(latitude, longitude, variable, mask,
                       interpolation_power, interpolation_smoothing,
                       cmap, title, title_size, legend, legend_size, colorbar_size):
    
    # Garante tipos numéricos
    latitude = pd.to_numeric(latitude, errors='coerce')
    longitude = pd.to_numeric(longitude, errors='coerce')
    variable = pd.to_numeric(variable, errors='coerce')

    # Cria DataFrame e remove valores inválidos
    df = pd.DataFrame({'latitude': latitude, 'longitude': longitude, 'data': variable})
    df = df.dropna(subset=['latitude', 'longitude', 'data'])

    # Constrói GeoDataFrame
    df['geometry'] = [Point(xy) for xy in zip(df['longitude'], df['latitude'])]
    gdf_points = gpd.GeoDataFrame(df, geometry='geometry', crs="EPSG:4326")

    # Grade de interpolação
    resolution = 0.01  # grau (~1km)
    minx, miny, maxx, maxy = gdf_points.total_bounds
    grid_x, grid_y = np.meshgrid(
        np.arange(minx, maxx, resolution),
        np.arange(miny, maxy, resolution)[::-1]
    )
    grid_coords = np.vstack((grid_x.ravel(), grid_y.ravel())).T

    # Interpolação IDW
    known_coords = np.array(list(zip(df['longitude'], df['latitude'])))
    known_values = df['data'].values
    tree = BallTree(known_coords, leaf_size=15)
    k = len(known_coords)
    dist, idx = tree.query(grid_coords, k=k)

    dist += max(interpolation_smoothing, 1e-10)  # evita divisão por zero
    weights = 1 / (dist ** interpolation_power)
    weights /= weights.sum(axis=1)[:, None]

    interpolated_grid = np.sum(known_values[idx] * weights, axis=1)
    interpolated_grid = interpolated_grid.reshape(grid_x.shape).astype(float)

    # Máscara shapefile
    gdf_mask = gpd.read_file(mask).to_crs("EPSG:4326")
    mask_array = features.rasterize(
        [(geom, 1) for geom in gdf_mask.geometry],
        out_shape=interpolated_grid.shape,
        transform=from_origin(minx, maxy, resolution, resolution),
        fill=0,
        dtype=np.uint8
    )
    interpolated_grid = np.where(mask_array == 1, interpolated_grid, np.nan)

    # Plot
    fig, ax = plt.subplots(figsize=(10, 10))
    extent = [minx, maxx, miny, maxy]
    cax = ax.imshow(interpolated_grid, extent=extent, origin='upper', cmap=cmap)
    gdf_mask.boundary.plot(ax=ax, color='black', linewidth=2.5)

    ax.set_title(title, fontsize=title_size, pad=20)
    ax.set_axis_off()

    if legend and legend_size > 0:
        cbar = plt.colorbar(cax, ax=ax, shrink=colorbar_size / 100, pad=0.02)
        cbar.set_label(legend, fontsize=legend_size, labelpad=20, rotation=90)

    plt.show()
    fig.savefig('interpolated_map.png', bbox_inches='tight', pad_inches=0.1)


def generate_colormap_image(cmap_name):
    gradient = np.linspace(0, 1, 256).reshape(1, -1)
    fig, ax = plt.subplots(figsize=(3, 0.5))
    ax.imshow(gradient, aspect='auto', cmap=cmap_name)
    ax.set_axis_off()
    buf = BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight', pad_inches=0)
    plt.close(fig)
    buf.seek(0)
    return buf

def main():
    st.set_page_config(layout="wide")
    st.markdown(
        body="""
            <style>
                .block-container{
                        padding-top: 25px;
                    }
            </style>
        """, 
        unsafe_allow_html=True
    )

    col1, col2 = st.columns([3, 2])
    st.title("Utilitário Automatizado de Interpolação (UAI)")


    if "ready" not in st.session_state:
        st.session_state.ready = False

    with st.container():
        # Layout de colunas (mapa mais largo que o gráfico)
        col1, col2 = st.columns([4, 2])  # proporção 4:2 para mapa ficar dominante
        with col1:
            # Seletor sobre o mapa
            data_file = st.file_uploader(
                "📂 Escolha um arquivo CSV ou Excel", 
                type=["csv", "xlsx"],
                label_visibility="visible",
                help="Carregue um arquivo de dados georreferenciados a ser interpolado.",
            
                )

            sub_col1, sub_col2 = st.columns(2)

            with sub_col1:

                if data_file is not None:
                    st.success("Arquivo carregado com sucesso!")

                use_header_choice = st.toggle(
                    "A primeira linha contém os nomes das colunas?",
                    value=False,
                    key="use_header_toggle",
                    help="Selecione se a primeira linha do arquivo contém o cabeçalho. "
                )

                skiprows=st.text_input(
                    "Número de linhas a serem puladas (0 para nenhuma):",
                    value=0,
                    key="skiprows_input",
                    help="Número de linhas a serem puladas no início do arquivo (sem considerar o cabeçalho)."
                )

            
            with sub_col2:
                st.info(
                    "Para que o resultado da interpolação corresponda a todo o limite desejado é necessário que o arquivo de dados contenha pontos externos ao limite.",
                    icon="⚠️"
                )
                shapefile_data = st.toggle(
                    "Limite diferente de Minas Gerais?",
                    value=False,
                    key="teste",
                    help="Selecione se deseja utilizar um limite diferente de Minas Gerais para a interpolação. ⚠️ É necessário que os pontos do arquivo de dados sejam compatíveis com o limite selecionado.",
                )

                if shapefile_data:
                    st.markdown("Arquivos do tipo Shapefile (.shp) disponíveis no portal do [IBGE](https://www.ibge.gov.br/geociencias/organizacao-do-territorio/malhas-territoriais/15774-malhas.html).")
                    shapefile_path = st.file_uploader(
                        "Carregue um arquivo Shapefile (.shp)",
                        type=["shp"],
                        label_visibility="visible",
                        help="Carregue um arquivo Shapefile para definir o recorte da interpolação."
                    )

            header=0
            if use_header_choice:
                header=1
        

            if data_file is not None:
                
                if data_file.name.endswith('.csv'):
                    df = pd.read_csv(data_file,skiprows=header)
                    df = df.iloc[int(skiprows):]
                elif data_file.name.endswith('.xlsx'):
                    df = pd.read_excel(data_file, skiprows=header)
                    df = df.iloc[int(skiprows):]

                
                st.dataframe(df, height=500)  

                
    with col2:
        st.markdown("### Parâmetros de Interpolação",
            help="⚠️Atenção!! Os valores de latitude e longitude devem estar em graus, com os decimais sepadados por ponto e o valor a ser interpolado deve ser numérico.",)
        if data_file is not None:
            latitude = st.selectbox(
                "Selecione a coluna da Latitude",
                options=df.columns.tolist(),
                key="latitude_select",
                    
            )
            longitude = st.selectbox(
                "Selecione a coluna da Longitude",
                options=df.columns.tolist(),
                key="longitude_select"
            )
            variable = st.selectbox(
                "Selecione o valor a ser interpolado",
                options=df.columns.tolist(),
                key="variable_select"
            )
            title= st.text_input(
                "Título do Mapa",
                value="",
                key="title_input",
                placeholder="Digite o título do gráfico"
            )
            title_size= st.slider(
                "Tamanho do Título",
                min_value=0,
                max_value=80,
                value=24,
                key="title_size_slider"
            )
            legend= st.text_input(
                "Legenda do Mapa",
                value="",
                key="legend_input",
                placeholder="Digite a legenda do gráfico"
            )

            col_01, col_02 = st.columns(2)
            with col_01:
                legend_size= st.slider(
                    "Tamanho da Legenda",
                    min_value=0,
                    max_value=40,
                    value=24,
                    key="legend_size_slider"
                )

                interpolation_intensity= st.slider(
                    "Intensidade da interpolação",
                    min_value=1,
                    max_value=10,
                    value=4,
                    step=1,
                    key="intensity_slider",
                    help="Peso atribuído a distância dos pontos (quanto menor, mais suave a interpolação)."  
                )

            with col_02:
                colorbar_size= st.slider(
                    "Tamanho da Barra de Cores",
                    min_value=0,
                    max_value=100,
                    value=50,
                    key="colorbar_size_slider"
                )

                interpolation_smoothing= st.slider(
                    "Suavização da Interpolação",
                    min_value=0.0,
                    max_value=1.0,
                    value=0.4,
                    step=0.01,
                    key="smoothing_slider",
                    help="Suavização da interpolação (quanto maior, mais suave será a transição entre os valores)."
                )




            if "colormap_custom" not in st.session_state:
                st.session_state.colormap_custom = False

            
            colormaps = ["viridis", "plasma", "inferno", "magma", "cividis", "Blues", "Greens", "Reds", "Purples", "Oranges",
                        "Greys", "YlOrRd", "YlGnBu", "YlGn", "BuPu", "GnBu", "PuBu", "OrRd", "PuRd", "RdPu",
                        "Blues_r", "Greens_r", "Reds_r", "Purples_r", "Oranges_r", "Greys_r"]

            st.session_state.selected_colormap = "viridis"

            with st.popover("🎨 Escolha o Mapa de Cores",use_container_width=True):
                st.markdown("Mapa de cores: Outros disponíveis no [Matplotlib](https://matplotlib.org/stable/tutorials/colors/colormaps.html).")

                for cmap in colormaps:
                    col1, col2 = st.columns([3, 1])
                    with col2:
                        if st.button(cmap, key=f"btn_{cmap}"):
                            st.session_state.selected_colormap = cmap   
                            st.session_state.colormap_custom = False

                    with col1:
                        st.image(generate_colormap_image(cmap), width=150, use_container_width=True)
                    st.divider()

                if not st.session_state.colormap_custom:
                    if st.button("Outro",use_container_width=True):
                        st.session_state.colormap_custom = True

                if st.session_state.colormap_custom:
                    other = st.text_input("Digite o nome do mapa de cores:", key="other_input")
                    if other:
                        st.session_state.selected_colormap = other

                if "selected_colormap" in st.session_state:
                    st.success(f"Mapa de cores selecionado: {st.session_state.selected_colormap}")



            interpolate_button = st.button("Interpolar", key="interpolate_button",
                                            help="Clique para gerar a interpolação com os parâmetros selecionados.",
                                            use_container_width=True,
                                            icon="🔍")

            
            if interpolate_button:
                st.success("Iniciando a interpolação...")
                with st.spinner("Processando (*musica de elevador*)...", show_time=True):
                    if data_file is not None and latitude and longitude and variable:
                        interpolation_func(
                            df[latitude],
                            df[longitude],
                            df[variable],
                            shapefile_path if shapefile_data else './mask/MG_UF_2024.shp',
                            interpolation_power=interpolation_intensity,
                            interpolation_smoothing=interpolation_smoothing,
                            cmap=st.session_state.selected_colormap,
                            title=title,
                            title_size=title_size,
                            legend=legend,
                            legend_size=legend_size,
                            colorbar_size=colorbar_size
                        )
                        st.session_state.ready= True
                        
                    
                        st.success("Interpolação concluída!")
                        st.balloons()
                    
                    else:
                        st.error("Por favor, preencha todos os campos necessários para a interpolação.")


        
    if st.session_state.ready:
        with open("interpolated_map.png", "rb") as f:
                    image_bytes = f.read()

        st.download_button(
            label="📥 Baixar imagem",
            data=image_bytes,
            file_name="interpolated_map.png",
            mime="image/png",
            
        )
        st.image('interpolated_map.png')


    st.divider()
    st.markdown("## ℹ️ Sobre")

    show_about = st.session_state.get("show_about", False)

    with st.expander("Clique para ver mais informações", expanded=show_about):
        st.markdown("""
        Esta aplicação foi desenvolvida para facilitar a visualização de dados georreferenciados, permitindo a interpolação sem a necessidade de conhecimento prévio em Python.
        
        **Uso e distribuição:** Esta aplicação pode ser usada e distribuída livremente, sinta-se a vontade para fazer modificações ou enviar sugestões!
        
        **Versão:** 1.0.0 \n
        **Data:** Julho de 2025\n
        
        **Sobre o desenvolvedor:** Sou estudante de ciência da computação na PUC-MG e acredito na livre cooperação entre mentes das
            mais diversas áreas para o desenvolvimento de soluções inovadoras. A análise de dados geograficamente foi fundamental 
            na minha área de estudo, sobre a previsão de irradiação solar com inteligência artificial, e torço para que também lhe seja útil. 
            Mesmo que uma única pessoa sequer sinta que essa ferramenta contribuiu de alguma forma em seus estudos, ela cumpriu o seu papel e o meu esfoço terá valido a pena!
        
        **Agradecimentos:** Agradecimentos especiais à minha orientadora, Cristiana Brasil Maia, obrigado por tudo 🌹.

        **Desenvolvido por:** Ricardo H. Guedes Furiati \n
        > Ad astra per aspera
        """)
        # Depois de mostrar, reseta o flag
        st.session_state.show_about = False


if __name__ == "__main__":
    main()
