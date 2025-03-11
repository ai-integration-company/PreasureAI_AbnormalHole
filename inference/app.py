import base64
import os
import numpy as np
import torch
from torch import nn
from torchvision import models, transforms
import cv2
from scipy.interpolate import interp1d
from PIL import Image
import streamlit as st
import pandas as pd

# ----------------------- Model & Helper Functions -----------------------

# Define feature names
bin_columns = [
    "Некачественное ГДИС",
    "Влияние ствола скважины",
    "Радиальный режим",
    "Линейный режим",
    "Билинейный режим",
    "Сферический режим",
    "Граница постоянного давления",
    "Граница непроницаемый разлом",
]
num_columns = [
    "Влияние ствола скважины_details",
    "Радиальный режим_details",
    "Линейный режим_details",
    "Билинейный режим_details",
    "Сферический режим_details",
    "Граница постоянного давления_details",
    "Граница непроницаемый разлом_details",
]

# Define model architecture
class MultiTaskResNet(nn.Module):
    def __init__(self, pretrained=False):
        super().__init__()
        self.base = models.resnet101(pretrained=pretrained)
        # Modify first conv layer for 6 channels
        orig_conv = self.base.conv1
        new_conv = nn.Conv2d(
            in_channels=6,
            out_channels=orig_conv.out_channels,
            kernel_size=orig_conv.kernel_size,
            stride=orig_conv.stride,
            padding=orig_conv.padding,
            bias=orig_conv.bias,
        )
        with torch.no_grad():
            new_conv.weight[:, :3, :, :] = orig_conv.weight[:, :3, :, :]
            new_conv.weight[:, 3:, :, :] = orig_conv.weight[:, :3, :, :]
        self.base.conv1 = new_conv

        num_feats = self.base.fc.in_features
        self.base.fc = nn.Identity()

        self.binary = nn.Sequential(
            nn.Linear(num_feats, 512),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(0.1),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(0.1),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(0.1),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(0.1),
            nn.Linear(64, 8),
        )

        self.resgresion = nn.Sequential(
            nn.Linear(num_feats + 24, 512),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(0.1),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(0.1),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(0.1),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(0.1),
            nn.Linear(64, 7),
        )

    def forward(self, x, start_values=None):
        out = self.base(x)
        logits_8 = self.binary(out)
        if start_values is None:
            regr_7 = torch.sigmoid(
                self.resgresion(torch.cat((out, torch.zeros(x.size()[0], 24, device=x.device)), axis=1))
            )
        else:
            regr_7 = torch.sigmoid(self.resgresion(torch.cat((out, start_values), axis=1)))
        return logits_8, regr_7

# Ranges for denormalization of regression outputs
RANGES = [
    (-0.5, 5.5),   # Влияние ствола скважины_details
    (-0.3, 4.1),   # Радиальный режим_details
    (-0.7, 2.2),   # Линейный режим_details
    (-0.4, 4.0),   # Билинейный режим_details
    (-0.3, 3.0),   # Сферический режим_details
    (2.4, 495.0),  # Граница постоянного давления_details
    (4.0, 555.0),  # Граница непроницаемый разлом_details
]

def denormalize_labels(pred_labels):
    pred_labels_orig = pred_labels.clone()
    for i in range(7):
        min_val, max_val = RANGES[i]
        pred_labels_orig[:, i] = pred_labels[:, i] * (max_val - min_val) + min_val
    return pred_labels_orig

# Create intermediate images from time series data
def create_and_save_graphics(time, pressure, devpressure, img_size=(256, 256, 3)):
    background = np.ones(img_size, dtype=np.uint8) * 255
    scatter_img = background.copy()
    plot_img = background.copy()
    
    if len(time) == 0:
        return plot_img, scatter_img

    time_min = min(time)
    time_max = max(time)
    pressure_min = min(pressure)
    pressure_max = max(pressure)
    devpressure_min = min(devpressure)
    devpressure_max = max(devpressure)

    max_y = max(devpressure_max, pressure_max)
    min_y = min(devpressure_min, pressure_min)

    grid_x = np.array([x / 2 for x in range(int(2 * time_min), int(2 * time_max) + 1)])
    grid_y = np.array([x / 2 for x in range(int(2 * min_y), int(2 * max_y) + 1)])

    if time_max - time_min == 0:
        norm_time = np.zeros(len(time), dtype=np.int32)
        grid_x_norm = np.zeros_like(grid_x, dtype=np.int32)
    else:
        norm_time = ((np.array(time) - time_min) / (time_max - time_min) * (img_size[1] - 10)).astype(np.int32)
        grid_x_norm = ((grid_x - time_min) / (time_max - time_min) * (img_size[1] - 10)).astype(np.int32)

    if max_y - min_y == 0:
        norm_pressure = np.zeros(len(pressure), dtype=np.int32)
        norm_devpressure = np.zeros(len(devpressure), dtype=np.int32)
        grid_y_norm = np.zeros_like(grid_y, dtype=np.int32)
    else:
        norm_pressure = ((np.array(pressure) - min_y) / (max_y - min_y) * (img_size[0] - 10)).astype(np.int32)
        norm_pressure = img_size[0] - norm_pressure
        norm_devpressure = ((np.array(devpressure) - min_y) / (max_y - min_y) * (img_size[0] - 10)).astype(np.int32)
        norm_devpressure = img_size[0] - norm_devpressure
        grid_y_norm = ((grid_y - min_y) / (max_y - min_y) * (img_size[0] - 10)).astype(np.int32)
        grid_y_norm = img_size[0] - grid_y_norm

    for x in grid_x_norm:
        cv2.line(scatter_img, (int(x), 0), (int(x), img_size[0]), color=(128, 128, 128), thickness=1)
        cv2.line(plot_img, (int(x), 0), (int(x), img_size[0]), color=(128, 128, 128), thickness=1)

    for y in grid_y_norm:
        cv2.line(scatter_img, (0, int(y)), (img_size[1], int(y)), color=(128, 128, 128), thickness=1)
        cv2.line(plot_img, (0, int(y)), (img_size[1], int(y)), color=(128, 128, 128), thickness=1)

    for i in range(len(time)):
        cv2.circle(scatter_img, (int(norm_time[i]), int(norm_pressure[i])), 1, (255, 0, 0), -1)
        cv2.circle(scatter_img, (int(norm_time[i]), int(norm_devpressure[i])), 1, (0, 0, 255), -1)

    for i in range(len(time) - 1):
        cv2.line(plot_img, (int(norm_time[i]), int(norm_pressure[i])),
                 (int(norm_time[i + 1]), int(norm_pressure[i + 1])), color=(255, 0, 0), thickness=1)
        cv2.line(plot_img, (int(norm_time[i]), int(norm_devpressure[i])),
                 (int(norm_time[i + 1]), int(norm_devpressure[i + 1])), color=(0, 0, 255), thickness=1)

    return plot_img, scatter_img

def process_sample(curr_time_series):
    time = np.array([triplet[0] for triplet in curr_time_series])
    pressure = np.array([triplet[1] for triplet in curr_time_series])
    derivative = np.array([triplet[2] for triplet in curr_time_series])
    if len(time) > 1:
        time = time - (min(time) - 1e-6)
        time = np.log(time)
        pressure = np.log(pressure)
        devpressure = np.log(derivative)
        time_interp = np.linspace(time.min(), time.max(), 10)
        interp_pressure = interp1d(time, pressure, kind="linear", fill_value="extrapolate")
        interp_devpressure = interp1d(time, devpressure, kind="linear", fill_value="extrapolate")
        pressure_interp = interp_pressure(time_interp)
        devpressure_interp = interp_devpressure(time_interp)
        plot_img, scatter_img = create_and_save_graphics(time, pressure, devpressure)
        np_labels = np.array(
            [time[0], time[-1], pressure[0], devpressure[0]] +
            list(pressure_interp) +
            list(devpressure_interp)
        )
    else:
        plot_img, scatter_img = create_and_save_graphics([], [], [])
        np_labels = np.array([0, 0, 0, 0] + 20 * [0])
    return plot_img, scatter_img, np_labels

# Image transformation for model input
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor()
])

# ----------------------- UI & Inference Integration -----------------------

st.markdown(f"""
<style>
/* Title (h1) should be #30d5c8 */
h1 {{
    color: #30d5c8 !important;
}}
</style>
""", unsafe_allow_html=True)


st.title("PressureAI")
st.markdown(
    """
    <div class="explanation">
      Вставьте csv файл с тремя временными рядами в формате обучающих данных.
    </div>
    """,
    unsafe_allow_html=True
)



uploaded_file = st.file_uploader("Выберите csv файл")
if uploaded_file is not None:
    file_uuid = uploaded_file.name  # Use filename as UUID
    try:
        content = uploaded_file.read().decode("utf-8").splitlines()
        curr_time_series = [tuple(map(float, line.strip().split("\t"))) for line in content if line.strip() != ""]
        st.success(f"Файл загружен успешно. UUID: {file_uuid}. Строк: {len(curr_time_series)}")
    except Exception as e:
        st.error("Ошибка при чтении файла: " + str(e))
        curr_time_series = None
    if curr_time_series:
        if st.button("Выполнить inference"):
            with st.spinner("Обработка..."):
                # Process sample to generate intermediate images and start values
                plot_img, scatter_img, np_labels = process_sample(curr_time_series)
                # Convert images from BGR to RGB for display
                plot_img_rgb = cv2.cvtColor(plot_img, cv2.COLOR_BGR2RGB)
                scatter_img_rgb = cv2.cvtColor(scatter_img, cv2.COLOR_BGR2RGB)
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.image(plot_img_rgb, caption="Plot Image (График)", width=250)

                with col2:
                    st.image(scatter_img_rgb, caption="Scatter Image (Разброс точек)", width=250)
                # Prepare model input
                plot_tensor = transform(Image.fromarray(plot_img_rgb))
                scatter_tensor = transform(Image.fromarray(scatter_img_rgb))
                res_img = torch.cat([plot_tensor, scatter_tensor], dim=0).unsqueeze(0)
                start_values = torch.tensor(np_labels).unsqueeze(0)
                # Load model (cached)
                @st.cache_resource
                def load_model():
                    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                    model = MultiTaskResNet()
                    state_dict_path = os.path.join("resnet101_0.9528.pth")
                    state_dict = torch.load(state_dict_path, map_location=device)
                    model.load_state_dict(state_dict)
                    model.to(device)
                    model.eval()
                    return model, device
                model, device = load_model()
                res_img = res_img.to(device).float()
                start_values = start_values.to(device).float()
                with torch.no_grad():
                    logits_8, regr_7 = model(res_img, start_values)
                    pred_binary = torch.sigmoid(logits_8).cpu().numpy().flatten()
                    pred_regression = regr_7.cpu().numpy()
                    pred_regression = denormalize_labels(torch.tensor(pred_regression)).cpu().numpy().flatten()
                    pred_binary = (pred_binary >= 0.5).astype(int)
                # Build full output for CSV download
                output_data = {"file": file_uuid}
                for name, val in zip(bin_columns, pred_binary):
                    output_data[name] = val
                for name, val in zip(num_columns, pred_regression):
                    output_data[name] = val
                df_full = pd.DataFrame([output_data])
                csv_full = df_full.to_csv(index=False, header=True)
                # Build display table: include only columns where corresponding binary equals 1.
                display_dict = {}
                for idx, feature in enumerate(num_columns):
                    if pred_binary[idx+1] == 1:
                        display_dict[feature] = [pred_regression[idx]]
                if not display_dict:
                    display_dict["Нет численных признаков"] = [""]
                df_display = pd.DataFrame(display_dict)
                st.write("**Результаты (численные значения для признаков, где бинарное = 1):**")
                st.table(df_display)
                st.download_button("Скачать CSV", csv_full, file_name="results.csv", mime="text/csv")
