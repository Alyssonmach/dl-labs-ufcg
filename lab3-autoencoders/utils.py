from tensorflow.keras import models
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import numpy as np
import os

# remove avisos do TensorFlow no console
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

def visualize_feature_maps(model, image_input):
    # acompanha o processo de remoção de ruído das imagens em todas as camadas.
    # gera visualizações dos mapas de características (filtros).

    # identifica camadas de interesse
    conv_outputs = [layer.output for layer in model.layers if "conv" in layer.name]
    conv_layer_names = [layer.name for layer in model.layers if "conv" in layer.name]

    if not conv_layer_names:
        print("Nenhuma camada convolucional encontrada para visualizar.")
        return

    # cria modelo de inspeção
    visualization_model = models.Model(inputs=model.input, outputs=conv_outputs)
    
    img_tensor = np.expand_dims(image_input, axis=0)
    
    # chamada direta do predict
    activations = visualization_model.predict(img_tensor, verbose=0)

    images_per_row = 16

    for layer_name, layer_activation in zip(conv_layer_names, activations):
        num_filters = layer_activation.shape[-1]
        spatial_size = layer_activation.shape[1]

        # lógica para definir linhas e colunas do grid
        num_cols = min(num_filters, images_per_row)
        num_rows = (num_cols // images_per_row) + 1 if num_cols % images_per_row != 0 else num_cols // images_per_row

        feature_grid = np.zeros((spatial_size * num_rows, spatial_size * num_cols))
            
        for col in range(num_cols):
            channel_image = layer_activation[0, :, :, col]
        
            # normalização para visualização
            if channel_image.std() > 0:
                channel_image -= channel_image.mean()
                channel_image /= (channel_image.std() + 1e-5)
                channel_image *= 64
                channel_image += 128
                channel_image = np.clip(channel_image, 0, 255).astype("uint8")
            else:
                channel_image = np.clip(channel_image, 0, 255).astype("uint8")

            row_idx = col // images_per_row
            col_idx = col % images_per_row
            
            feature_grid[
                row_idx * spatial_size : (row_idx + 1) * spatial_size,
                col_idx * spatial_size : (col_idx + 1) * spatial_size
            ] = channel_image

        scale = 1.0 / spatial_size
        plt.figure(figsize=(scale * feature_grid.shape[1] * 2.5, 
                            scale * feature_grid.shape[0] * 2.5))
        
        layer_shape_info = layer_activation.shape[1:]
        plt.title(
            f"Camada: {layer_name}\nSaída: {layer_shape_info} (Altura, Largura, Canais)",
            fontsize=14,
            pad=10
        )
        
        plt.grid(False)
        plt.imshow(feature_grid, aspect="equal", cmap="viridis")
        plt.xticks([])
        plt.yticks([])
        plt.tight_layout()
        plt.show()

def visualize_architecture_final(model, input_image):
    # gera uma visualização 3D interativa da arquitetura do autoencoder.
    # mostra o fluxo da esquerda para a direita, com destaque para a compressão no bottleneck.

    # extrai as ativações
    layer_outputs = [layer.output for layer in model.layers if "conv" in layer.name]
    layer_names = [layer.name for layer in model.layers if "conv" in layer.name]
    
    activation_model = models.Model(inputs=model.input, outputs=layer_outputs)
    img_tensor = np.expand_dims(input_image, axis=0)
    activations = activation_model.predict(img_tensor, verbose=0)

    fig = go.Figure()
    
    x_spacing = 40 
    current_x = 0
    
    layer_coords = []

    input_display = input_image.squeeze()
    input_display = np.flipud(input_display)
    h, w = input_display.shape
    
    y_start, y_end = -w/2, w/2
    z_start, z_end = -h/2, h/2
    
    layer_coords.append((current_x, y_start, y_end, z_start, z_end))

    y_range = np.linspace(y_start, y_end, w)
    z_range = np.linspace(z_start, z_end, h)
    y_grid, z_grid = np.meshgrid(y_range, z_range)
    x_grid = np.full_like(y_grid, current_x)
    
    # superfície do input
    fig.add_trace(go.Surface(
        x=x_grid, y=y_grid, z=z_grid,
        surfacecolor=input_display,
        colorscale="gray",
        showscale=False,
        opacity=0.9,
        name="Input Raw",
        hoverinfo="name"
    ))
    
    # rótulo do input
    fig.add_trace(go.Scatter3d(
        x=[current_x], y=[0], z=[z_start - 3], 
        mode="text", text=[f"{h}x{w}"],
        textposition="bottom center",
        textfont=dict(size=10, color="white"),
        hoverinfo="skip"
    ))
    
    current_x += x_spacing

    for i, (layer_name, activation) in enumerate(zip(layer_names, activations)):
        layer_map = np.mean(activation[0], axis=-1)
        layer_map = np.flipud(layer_map)
        
        layer_map = (layer_map - layer_map.min()) / (layer_map.max() - layer_map.min() + 1e-5)
        
        h_layer, w_layer = layer_map.shape
        
        y_start, y_end = -w_layer/2, w_layer/2
        z_start, z_end = -h_layer/2, h_layer/2
        
        layer_coords.append((current_x, y_start, y_end, z_start, z_end))
        
        y_range = np.linspace(y_start, y_end, w_layer)
        z_range = np.linspace(z_start, z_end, h_layer)
        y_grid, z_grid = np.meshgrid(y_range, z_range)
        x_grid = np.full_like(y_grid, current_x)
        
        if h_layer == 7: 
            colorscale = "hot" 
            opacity = 1.0
        elif "transpose" in layer_name:
            colorscale = "electric" 
            opacity = 0.85
        else:
            colorscale = "viridis"
            opacity = 0.85

        fig.add_trace(go.Surface(
            x=x_grid, y=y_grid, z=z_grid,
            surfacecolor=layer_map,
            colorscale=colorscale,
            showscale=False,
            opacity=opacity,
            name=layer_name,
            hoverinfo="name+z+y"
        ))
        
        fig.add_trace(go.Scatter3d(
            x=[current_x]*5,
            y=[y_start, y_end, y_end, y_start, y_start],
            z=[z_start, z_start, z_end, z_end, z_start],
            mode="lines",
            line=dict(color="white", width=2),
            hoverinfo="skip"
        ))
        
        # rótulo
        fig.add_trace(go.Scatter3d(
            x=[current_x], y=[0], z=[z_start - 3], 
            mode="text", text=[f"{h_layer}x{w_layer}"],
            textposition="bottom center",
            textfont=dict(size=10, color="white"),
            hoverinfo="skip"
        ))

        current_x += x_spacing

    for i in range(len(layer_coords) - 1):
        curr = layer_coords[i]
        next_l = layer_coords[i+1]
        
        xs = [curr[0], next_l[0]]
        
        corners_curr = [
            (curr[1], curr[3]), (curr[2], curr[3]), 
            (curr[2], curr[4]), (curr[1], curr[4])
        ]
        corners_next = [
            (next_l[1], next_l[3]), (next_l[2], next_l[3]), 
            (next_l[2], next_l[4]), (next_l[1], next_l[4])
        ]
        
        for c_curr, c_next in zip(corners_curr, corners_next):
            fig.add_trace(go.Scatter3d(
                x=xs,
                y=[c_curr[0], c_next[0]],
                z=[c_curr[1], c_next[1]],
                mode="lines",
                line=dict(color="rgba(255, 255, 255, 0.3)", width=1, dash="dot"),
                hoverinfo="skip"
            ))

    fig.update_layout(
        title="Fluxo Arquitetural",
        title_font_color="white",
        showlegend=False,
        width=1000, height=600,
        template="plotly_dark",
        scene=dict(
            xaxis=dict(visible=False),
            yaxis=dict(visible=False),
            zaxis=dict(visible=False),
            aspectratio=dict(x=3, y=1, z=1),
            camera=dict(eye=dict(x=1.5, y=-2.0, z=1.0)),
            bgcolor="rgba(0,0,0,0)"
        ),
        margin=dict(l=0, r=0, b=0, t=50)
    )
    
    fig.show()