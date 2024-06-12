import trimap
import pacmap
import pandas as pd
import streamlit as st
from datetime import datetime
import streamlit as st
from sklearn.metrics import confusion_matrix, accuracy_score
from pickle import FALSE
from streamlit_option_menu import option_menu
import streamlit as st
from io import BytesIO
import umap.umap_ as umap
import plotly.express as px
from datetime import datetime
import trimap

from sklearn.manifold import TSNE
import plotly.express as px
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import numpy as np
import plotly.express as px
# Set the page config
st.set_page_config(page_title='Dimensionality Reduction Techniques', layout='wide', initial_sidebar_state='expanded')

# Title
st.title('📊 Data Visualizer')

with st.sidebar:
    selected = option_menu(None, ["Home", "Upload", "Visualizations", "Contact"],
                           icons=["house", "upload", "list-task", "envelope"], menu_icon="cast", default_index=0)

if 'plots' not in st.session_state:
    st.session_state.plots = {}

# Content based on sidebar selection

if selected == "Home":
    st.header("About the Application")
    st.markdown("""
        <div style="text-align: justify;">
            The application enables easy and fast visualization of complex data sets, offering our users a rich set of tools for analysis and information presentation. Users can work with data in various formats, utilizing advanced dimensionality reduction techniques such as UMAP, t-SNE, PAcMAP, and TRIMAP, which allow for effective and intuitive exploration of structures hidden within large data sets. The application also supports interactive elements, such as file selection, column selection for analysis, and choosing the type of chart, enabling dynamic manipulation and customization of visualizations to meet individual user needs.
        </div>
        """, unsafe_allow_html=True)

    with st.expander("UMAP"):
        st.write(
            """UMAP is a dimensionality reduction technique that aims to map high-dimensional data to a lower-dimensional space, typically two or three dimensions. It is a non-linear technique that works by finding low-dimensional representations that preserve local data structures. UMAP is a relatively new method that has become popular due to its ability to preserve both global and local data structures.""")
    with st.expander("T-SNE"):
        st.write(
            """T-SNE is a dimensionality reduction technique also used for mapping high-dimensional data to a lower-dimensional space. Like UMAP, t-SNE is non-linear and aims to preserve the structure of the data. Its main strength lies in its ability to detect complex structures and clusters in data, making it a popular tool in visual and exploratory data analysis.""")
    with st.expander("PAcMAP"):
        st.write(
            """PAcMAP is a dimensionality reduction method that utilizes the Maximum A Posteriori Probability (MAP) technique to project data into a lower-dimensional space. It is a probabilistic method that attempts to retain the most important information about the data while reducing dimensions. PAcMAP is a relatively new method that builds upon the development of probabilistic-based dimensionality reduction techniques.""")
    with st.expander("TriMAP"):
        st.write(
            """TRIMAP is a dimensionality reduction technique that works by mapping data onto a triangular structure in lower dimensionality. It is a method that aims to preserve distances and data structure by representing them in the form of triangles. TRIMAP can be applied to various datasets, but it is particularly effective for high-dimensional data with complex structure.""")


def string_to_numeric_data(df):
    dct = {}
    for column in df.select_dtypes(include=[object]).columns:
        df[column], mapping = pd.factorize(df[column])
        dct[column] = {index: label for index, label in enumerate(mapping)}
    return df, dct

# Validator to check uploaded file format
def validate_file(uploaded_file):
    if not uploaded_file:
        st.error("No file uploaded.")
        return False
    if not uploaded_file.name.endswith(('.csv', '.xlsx', '.json')):
        st.error("Unsupported file format. Please upload a CSV, XLSX, or JSON file.")
        return False
    return True

@st.cache_data()
def load_and_process_data(uploaded_file):
    if uploaded_file.name.endswith('.csv'):
        df = pd.read_csv(uploaded_file)
    elif uploaded_file.name.endswith('.xlsx'):
        df = pd.read_excel(uploaded_file)
    elif uploaded_file.name.endswith('.json'):
        df = pd.read_json(uploaded_file)

    df, dct = string_to_numeric_data(df)
    df = df.fillna(0)  # Fill NaN values with 0
    return df, dct

# Upload section
if selected == "Upload":
    uploaded_file = st.file_uploader("Drop or select your file", type=['csv', 'xlsx', 'json'])
    if validate_file(uploaded_file):
        try:      
            df, dct = load_and_process_data(uploaded_file)
            columns = df.columns.tolist()
            selected_columns = st.multiselect('Select columns to display:', columns, default=columns)
            df_selected = df[selected_columns]
            st.dataframe(df_selected, width=1500)

            if dct:
                column_to_view = st.selectbox("Choose a column to see its legend:", list(dct.keys()),
                                            help="Select a column to view the mapping of numerical values back to their original string representations.")
                if column_to_view:
                    with st.expander(f"Legend for '{column_to_view}':"):
                        mapping_dict = dct[column_to_view]
                        for num, name in mapping_dict.items():
                            st.write(f"{num} : {name}")

            # label_column = st.selectbox("Select the label column:", options=columns,
                                    #   help="Select the column that contains labels for the data.")
            # to etykietowanie ;(((((

            # Scaling options
            scale_option = st.selectbox(
                "Choose a scaling method:",
                ['None', 'Min-Max Scaling', 'Standard Scaling'],
                help="Select how to scale your data."
            )

            if scale_option == 'Min-Max Scaling':
                scaler = MinMaxScaler()
                numeric_columns = df_selected.select_dtypes(include=[np.number]).columns.tolist()
                df_selected[numeric_columns] = scaler.fit_transform(df_selected[numeric_columns])
            elif scale_option == 'Standard Scaling':
                scaler = StandardScaler()
                numeric_columns = df_selected.select_dtypes(include=[np.number]).columns.tolist()
                df_selected[numeric_columns] = scaler.fit_transform(df_selected[numeric_columns])

            # Download button for the processed data
            excel_data = BytesIO()
            df_selected.to_excel(excel_data, index=False)
            excel_data.seek(0)
            st.download_button("Download Excel", excel_data, "selected_data.xlsx",
                            "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")


            # Visualization options
            plot_types = st.multiselect(
                "Choose the method types:",
                ['UMAP', 'T-SNE', 'TriMap', 'PaCMAP'],
                help="Select the methods for data visualization."
            )

            size = st.select_slider(
                'Select the size of visualization data',
                options=['1k', '2k', '5k', '10k', '20k'],
                help="Choose how many data points to include in the visualization."
            )
            st.caption(f"Selected size: {size}")

            color1 = st.color_picker('Select color for component 1', '#755EFF')
            color2 = st.color_picker('Select color for component 2', '#FF5733')
            color3 = st.color_picker('Select color for component 3', '#33FF57')

            if plot_types:
                for plot_type in plot_types:
                    st.subheader(f"{plot_type} projection")
                    with st.container():
                        col1, col2 = st.columns([1, 3])
                        with col1:
                            n_components = st.slider('Number of components', 2, 3, 2, key=f'{plot_type}_components')
                            if plot_type == 'UMAP':
                                n_neighbors = st.slider('Number of neighbors', 5, 50, 15, key='umap_neighbors')
                                min_dist = st.slider('Minimum distance', 0.1, 0.9, 0.1, step=0.1, key='umap_min_dist')
                                reducer = umap.UMAP(n_neighbors=n_neighbors, min_dist=min_dist, n_components=n_components,
                                                    random_state=42)
                                embedding = reducer.fit_transform(df_selected)
                            elif plot_type == 'T-SNE':
                                perplexity = st.slider('Perplexity', 5, 50, 30, key='tsne_perplexity')
                                tsne = TSNE(n_components=n_components, perplexity=perplexity, random_state=42)
                                embedding = tsne.fit_transform(df_selected)
                            elif plot_type == 'PaCMAP':
                                n_neighbors = st.slider('Number of neighbors', 3, 100, 50, key='pacmap_neighbors')
                                pmap = pacmap.PaCMAP(n_neighbors=n_neighbors, n_components=n_components)
                                embedding = pmap.fit_transform(df_selected.values)
                            elif plot_type == 'TriMap':
                                n_neighbors = st.slider('Number of neighbors', 10, 100, 40, key='trimap_neighbors')
                                tmap = trimap.TRIMAP(n_iters=400, n_inliers=n_neighbors)
                                embedding = tmap.fit_transform(df_selected.values)

                        df_plot = pd.DataFrame(embedding, columns=[f'Component {i + 1}' for i in range(n_components)])

                        if n_components == 3:
                            df_plot['Component'] = pd.cut(df_plot.iloc[:, 0], bins=3, labels=['1', '2', '3'])
                            color_map = {'1': color1, '2': color2, '3': color3}
                            fig = px.scatter_3d(df_plot, x='Component 1', y='Component 2', z='Component 3',
                                                color='Component', color_discrete_map=color_map,
                                                title=f'{plot_type} projection', width=600, height=600)
                            # fig.update_layout(width=600, height=600)  # Ensure the plot is square
                        else:
                            df_plot['Component'] = pd.cut(df_plot.iloc[:, 0], bins=2, labels=['1', '2'])
                            color_map = {'1': color1, '2': color2}
                            fig = px.scatter(df_plot, x='Component 1', y='Component 2', color='Component',
                                            color_discrete_map=color_map, title=f'{plot_type} projection',
                                            width=600, height=600)
                            fig.update_layout(
                                margin=dict(l=20, r=20, b=20, t=80),
                                xaxis=dict(
                                    scaleanchor="y",
                                    scaleratio=1,
                                    constrain="domain"
                                ),
                                yaxis=dict(
                                    scaleanchor="x",
                                    scaleratio=1,
                                    constrain="domain"
                                )
                            )

                        with col2:
                            st.plotly_chart(fig, use_container_width=True)

                            # Add confusion matrix and accuracy score
                            if n_components == 3:
                                y_true = np.random.randint(0, 3, len(df_selected))  # Dummy true labels
                                y_pred = np.random.randint(0, 3, len(df_selected))  # Dummy predicted labels
                            else:
                                y_true = np.random.randint(0, 2, len(df_selected))  # Dummy true labels
                                y_pred = np.random.randint(0, 2, len(df_selected))  # Dummy predicted labels
                            cm = confusion_matrix(y_true, y_pred)
                            accuracy = accuracy_score(y_true, y_pred)

                            st.markdown("**Confusion Matrix for {}**".format(plot_type), unsafe_allow_html=True)
                            st.write(cm, use_container_width=True)

                            st.markdown("**Accuracy for {}**".format(plot_type), unsafe_allow_html=True)
                            st.write(accuracy)

                        if st.button(f"Save {plot_type} Plot"):
                            plot_key = f"{plot_type}_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"
                            st.session_state.plots[plot_key] = fig
                            st.success(f'{plot_type} Plot saved successfully!')

        except Exception as e:
            st.error(f"An error occurred: {e}")


# Visualizations section
elif selected == "Visualizations":
    st.header("Saved Plots")
    for plot_key in sorted(st.session_state.plots.keys(), reverse=True):
        with st.expander(f"Plot created on {plot_key.split('_')[1]}"):
            st.plotly_chart(st.session_state.plots[plot_key], use_container_width=True)

elif selected == "Contact":
    st.header("Contact with us:")

    contacts = [
        {
            "name": "Dariusz",
            "role": "Developer",
            "email": "dariusz.developer@gmail.com",
            "photo": "darek1.png"
        },
        {
            "name": "Aleksandra",
            "role": "UX/UI Designer",
            "email": "aleksandra.designer@gmail.com",
            "photo": "ola3.png"
        },
        {
            "name": "Klaudia",
            "role": "PM",
            "email": "klaudia.pm@gmail.com",
            "photo": "klaudia2.png"
        }
    ]

    # Display the contacts
    cols = st.columns(len(contacts))
    for idx, contact in enumerate(contacts):
        with cols[idx]:
            st.image(contact["photo"], width=200, use_column_width=True)

            # HTML bo nie umiem inaczej wyśrodkować tekstu xd
            html_str = f"""
                <div style="text-align: center;">
                    <h3>{contact["name"]}</h3>
                    <p>{contact["role"]}</p>
                    <button onclick="window.location.href='mailto:{contact["email"]}';" style="display: inline-block; background-color: #FF00A1; color: white; padding: 8px 12px; text-align: center; border: none; border-radius: 4px; cursor: pointer;">
                        Email {contact["name"]}
                    </button>
                </div>
                """
            st.markdown(html_str, unsafe_allow_html=True)

    # def show_email_address(address):
    #   if st.button("Email Us"):
    #       st.write(f"To: {address}")
    # show_email_address(email_address)
    # st.header("Contact Details")
    # st.write("Contact Us at [email@example.com](mailto:email@example.com)")
    # office_address = "[Office - University of Science and Technology in Cracow, Building D17](https://www.informatyka.agh.edu.pl/pl/wydzial/kontakt/)"

    # def show_office_address(address):

    #   if st.button("Office Address"):
    #      st.markdown(address, unsafe_allow_html=True)
    # show_office_address(office_address)

