"""Interfaz para procesamiento de señales."""

import os

import matplotlib.pyplot as plt
import numpy as np
from pydub import AudioSegment
from PyQt5.QtCore import QSize, Qt, QUrl
from PyQt5.QtGui import QFont, QIcon, QPixmap
from PyQt5.QtMultimedia import QMediaContent, QMediaPlayer
from PyQt5.QtWidgets import (
    QApplication,
    QComboBox,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QSlider,
    QSpinBox,
    QTabWidget,
    QVBoxLayout,
    QWidget,
    QSizePolicy,
)
from scipy.io import wavfile
from scipy.signal import filtfilt, firwin, iirfilter, kaiserord, lfilter


class MyWindow(QWidget):
    """Creacion de Interfaz."""

    def __init__(self):
        """
        Initialize the interface design.
        """
        super().__init__()
        self.setGeometry(
            0, 0, 2300, 1400
        ) 
        self.setWindowTitle("HMI")
        self.media_player = QMediaPlayer()
        self.media_player.setVolume(50)
        self.sampFreq = 4410  # Establecer frecuencia de muestreo inicial
        self.sound = None
        self.main_lay()

    def main_lay(self):
        """
        Set up the layout of the user interface.
        """

        layout = QVBoxLayout()  # Create a vertical layout

        # Create a main label for the interface with specific text and styling
        self.mainlabel = QLabel("HMI para procesamiento de señales.")
        font = QFont("Arial", 18, QFont.Bold)  
        self.mainlabel.setFont(font)  # Set the font for the label
        self.mainlabel.setAlignment(Qt.AlignLeft)  # Align the label text to the left
        layout.addWidget(self.mainlabel)  # Add the label to the layout

        # Create a label for the author's name with specific styling
        self.name = QLabel("Armando Genis Alvarez A01654262")
        font2 = QFont("Arial", 12, QFont.Medium) 
        self.name.setFont(font2)  # Set the font for this label
        self.name.setAlignment(Qt.AlignLeft)  # Align the label text to the left
        layout.addWidget(self.name)  # Add the label to the layout

        # Create a tab widget for organizing different functionality
        self.tabs = QTabWidget()
        self.tabs.setTabPosition(QTabWidget.West)  # Set the position of tabs to the west side
        layout.addWidget(self.tabs)  # Add the tab widget to the layout

        # Setup the 'Load Audio' tab
        self.carga_de_audio = QWidget()
        self.carga_de_audioLayout()  # Setup the layout for the audio load tab
        self.tabs.addTab(self.carga_de_audio, "Cargar Audio")  # Add the tab to the tab widget

        # Setup the 'Fourier Transform' tab
        self.ft_tab = QWidget()
        self.ft_tabLayout()  # Setup the layout for the Fourier Transform tab
        self.tabs.addTab(self.ft_tab, "Transformada de Fourier")  # Add the tab to the tab widget

        # Setup the 'Audio Filtering' tab
        self.filtro_tab = QWidget()
        self.filtros_tabLayout()  # Setup the layout for the audio filtering tab
        self.tabs.addTab(self.filtro_tab, "Filtros para Audio")  # Add the tab to the tab widget

        self.setLayout(layout)  # Set the created layout as the main layout of the widget

    def carga_de_audioLayout(self):
        """
        Set up the layout for the 'Load Audio' tab in the user interface.
        """
        vertical_layout = QVBoxLayout()  # Main layout is a vertical layout

        # Create a horizontal layout to hold the button
        horizontal_layout = QHBoxLayout()
        self.audio_load_button = QPushButton("Cargar Audio")  # Create a button for loading audio files
        self.audio_load_button.clicked.connect(self.archivo_loaded)  # Connect button click to the loading function

        # Style the button with custom colors and rounded corners
        self.audio_load_button.setStyleSheet(
            "background-color : #20aee6; color: white; border-radius: 30px;"
        )
        self.audio_load_button.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)  # Button expands in width but fixed in height
        horizontal_layout.addWidget(self.audio_load_button)  # Add button to the horizontal layout
        self.audio_load_button.setFixedHeight(50)  # Set the fixed height of the button

        vertical_layout.addLayout(horizontal_layout)  # Add the horizontal layout containing the button to the vertical layout
        vertical_layout.addStretch(1)  # Adds a stretch space after the button, pushing it towards the top of the layout

        self.carga_de_audio.setLayout(vertical_layout)  # Set the vertical layout as the layout for the 'Load Audio' tab


    def archivo_loaded(self):
        """
        Handle the event when an audio file is loaded through the file dialog.
        """
        file_dialog = self.setup_file_dialog()  # Set up and retrieve a file dialog for selecting audio files

        if file_dialog.exec_():  # Execute the file dialog
            file_paths = file_dialog.selectedFiles()  # Get the list of selected file paths from the dialog
            if file_paths:  # Check if any file paths were selected
                file_path = file_paths[0]  # Take the first file path from the list
                self.process_audio_file(file_path)  # Process the selected audio file
                self.plot_audio_signal()  # Plot the audio signal of the processed file
                self.update_interface(file_path)  # Update the interface with details from the loaded file


    def setup_file_dialog(self):
        """Set up and return a configured file dialog."""
        file_dialog = QFileDialog()
        file_dialog.setNameFilter("Archivos de audio (*.wav *.mp3 *.aac)")
        file_dialog.selectNameFilter("Archivos de audio (*.wav *.mp3 *.aac)")
        file_dialog.setViewMode(QFileDialog.List)
        return file_dialog

    def process_audio_file(self, file_path):
        """Process the selected audio file based on its format."""
        if file_path.endswith(".wav"):
            self.load_wav_file(file_path)
        elif file_path.endswith(".mp3"):
            self.load_mp3_file(file_path)
        elif file_path.endswith(".aac"):
            self.load_aac_file(file_path)
        self.normalize_audio()

    def load_wav_file(self, file_path):
        """Load and process WAV file."""
        self.sampFreq, self.sound = wavfile.read(file_path)
        if self.sound.ndim == 2:  # Convert stereo to mono
            self.sound = np.mean(self.sound, axis=1)

    def load_mp3_file(self, file_path):
        """Load and process MP3 file."""
        temp = AudioSegment.from_mp3(file_path)
        self.convert_audio_segment(temp)

    def load_aac_file(self, file_path):
        """Load and process AAC file."""
        temp = AudioSegment.from_file(file_path, format="aac")
        self.convert_audio_segment(temp)

    def convert_audio_segment(self, audio_segment):
        """Convert audio segment to numpy array and set sample frequency."""
        if audio_segment.channels > 1:
            audio_segment = audio_segment.set_channels(1)
        self.sound = np.array(audio_segment.get_array_of_samples())
        self.sampFreq = audio_segment.frame_rate

    def normalize_audio(self):
        """Normalize the audio signal to fit within 16-bit range."""
        self.sound = self.sound / 2.0**15

    def plot_audio_signal(self):
        """Plot and save the audio signal."""
        plt.plot(self.sound[:], "#20aee6")
        plt.xlabel("Señal de audio")
        plt.title("Señal original")
        plt.tight_layout()
        plt.savefig("senal_original.png")  # Save the plot of the audio signal
        plt.close()


    def update_interface(self, file_path):
        """
        Update the user interface to reflect the loaded audio file.
        """
        # Update the label with the name of the loaded file
        self.file_label = QLabel()
        self.file_label.setText(f"Archivo cargado: {os.path.basename(file_path)}")
        self.file_label.setAlignment(Qt.AlignCenter)  # Ensure the label text is centered
        font_for_file_label = QFont("Arial", 14, QFont.Bold)  # Using Arial, font size 14, bold
        self.file_label.setFont(font_for_file_label)

        carga_de_audio_layout = self.carga_de_audio.layout()
        carga_de_audio_layout.addWidget(self.file_label, 1, Qt.AlignCenter)  # Add the label at position 1, centered

        audio_image_layout = QHBoxLayout()
        audio_image_layout.setAlignment(Qt.AlignCenter)  # Align the entire horizontal layout to the center

        audio_image_widget = self.show_image("senal_original.png", 1200, 600)
        audio_image_layout.addWidget(audio_image_widget, 0, Qt.AlignCenter)  # Add the image widget centered in the layout

        carga_de_audio_layout.insertLayout(2, audio_image_layout, stretch=0)

        self.carga_de_audio.setLayout(carga_de_audio_layout)  # Apply the layout to 'carga_de_audio'


    def ft_tabLayout(self):
        """
        Set up the layout for the Fourier Transform tab in the user interface.
        """
        fourier_layout = QVBoxLayout()  # Main layout is a vertical layout

        # Design the Fourier Transform button
        self.fourier_button = QPushButton("Transformada de Fourier")  # Text directly in the constructor
        self.fourier_button.clicked.connect(self.fourier_transform)  # Connect to the Fourier transform function

        self.fourier_button.setIconSize(QSize(90, 90))  # Icon size, adjust as needed
        self.fourier_button.setStyleSheet(
            "background-color : #20aee6; color: white; border-radius: 30px;"  # Styling for the button
        )
        self.fourier_button.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)  # Button expands horizontally, fixed height
        self.fourier_button.setFixedHeight(50)  # Set the fixed height of the button

        fourier_layout.addWidget(self.fourier_button)  # Add button to layout without specifying alignment
        fourier_layout.addStretch(1)  # Add stretch after the button, which pushes it to the top

        self.ft_tab.setLayout(fourier_layout)  # Set the layout on the ft_tab


    def fourier_transform(self):
        """
        Apply Fourier Transform to the audio signal and plot the resulting spectrum.
        """
        # Compute the Fourier Transform of the audio signal
        fft_spectrum = np.fft.rfft(self.sound)  # Real FFT to handle real input signals
        freq = np.fft.rfftfreq(self.sound.size, d=1.0 / self.sampFreq)  # Frequency axis for plotting
        fft_spectrum_abs = np.abs(fft_spectrum)  # Magnitude of the FFT for displaying amplitude

        # Plotting the Fourier Transform spectrum
        plt.plot(freq, fft_spectrum_abs, "#20aee6")  # Use a distinctive blue color for the plot line
        plt.xlabel("frequency, Hz")  # Label for the x-axis
        plt.ylabel("Amplitude, units")  # Label for the y-axis
        plt.title("Transformada de Fourier")  # Title of the plot
        plt.tight_layout()  # Adjust layout to make it neat
        plt.savefig("ft.png")  # Save the Fourier Transform plot to a file
        plt.close()  # Close the plot to free up resources

        # Update the interface to reflect changes after the Fourier Transform
        self.update_fourier_interface()


    def update_fourier_interface(self):
        """
        Update the Fourier Transform tab interface to show the result of the Fourier analysis.
        """

        # Set the title label for the Fourier Transform tab
        self.ft_label = QLabel("Transformada de Fourier")
        self.ft_label.setAlignment(Qt.AlignCenter)  # Ensure the label text is centered
        ft_font = QFont("Arial", 14, QFont.Medium)  # Using Arial for a more modern look
        self.ft_label.setFont(ft_font)

        ft_layout = self.ft_tab.layout()
        ft_layout.addWidget(self.ft_label, 0, Qt.AlignCenter) 

        ft_image_layout = QVBoxLayout()
        ft_image_layout.setAlignment(Qt.AlignCenter)  

        # Create a widget to hold the image and add it to the image layout, centrally aligned
        ft_image_widget = self.show_image("ft.png", 1200, 600)
        ft_image_layout.addWidget(ft_image_widget, 0, Qt.AlignCenter)  # Add the image widget centered in the layout

        # Insert the image layout into the main layout below the label, ensuring it's centered
        ft_layout.addLayout(ft_image_layout, 1)  # Place the image layout immediately after the label

        self.ft_tab.setLayout(ft_layout)  # Apply the updated layout to 'ft_tab'



    def filtros_tabLayout(self):
        layout = QVBoxLayout()
        layout.setAlignment(Qt.AlignTop)

        # Choose Filter Type
        filter_type_layout = QHBoxLayout()
        filter_type_label = QLabel("Tipo de filtro:")
        font = QFont("Arial", 12, QFont.Bold)  # Changed to Arial for a modern look
        filter_type_label.setFont(font)
        self.filter_type = QComboBox()
        self.filter_type.addItems(["IIR", "FIR"])
        self.filter_type.setStyleSheet(
            "background-color: #337ab7; color: white; border-radius: 8px;"  # Updated color scheme
        )
        filter_type_layout.addWidget(filter_type_label)
        filter_type_layout.addWidget(self.filter_type)
        layout.addLayout(filter_type_layout)

        order_layout = QHBoxLayout()
        order_label = QLabel("Orden de filtro:")
        order_label.setFont(font)
        self.order_spinbox = QSpinBox()
        self.order_spinbox.setRange(1, 10)
        self.order_spinbox.setValue(4)
        self.order_spinbox.setStyleSheet(
            "background-color: #337ab7; color: white; border-radius: 8px;"  # Matching style
        )
        order_layout.addWidget(order_label)
        order_layout.addWidget(self.order_spinbox)
        layout.addLayout(order_layout)

        bandpass_layout = QHBoxLayout()
        bandpass_label = QLabel("Frecuencia:")
        bandpass_label.setFont(font)
        self.bandpass = QComboBox()
        self.bandpass.addItems(["lowpass", "highpass", "bandpass"])
        self.bandpass.setStyleSheet(
            "background-color: #337ab7; color: white; border-radius: 8px;"  # Consistent style
        )
        bandpass_layout.addWidget(bandpass_label)
        bandpass_layout.addWidget(self.bandpass)
        layout.addLayout(bandpass_layout)
        self.bandpass.currentIndexChanged.connect(self.updateBandpass)

        slider_layout = QHBoxLayout()
        self.parameter_edit = QLineEdit()
        self.parameter_edit.setFixedWidth(100)
        self.parameter_edit.setText("50")
        self.parameter_edit.setStyleSheet(
            "background-color: #337ab7; color: white; border-radius: 8px;"  # Unified style
        )
        slider_label = QLabel("Frec de corte 1:")
        slider_label.setFont(font)
        self.parameter_slider = QSlider(Qt.Horizontal)
        self.parameter_slider.setMinimum(1)
        self.parameter_slider.setMaximum(int(self.sampFreq / 2))
        self.parameter_slider.setValue(50)
        self.parameter_slider.setTickInterval(100)
        self.parameter_slider.setTickPosition(QSlider.TicksBelow)
        slider_layout.addWidget(slider_label)
        slider_layout.addWidget(self.parameter_edit)
        slider_layout.addWidget(self.parameter_slider)
        layout.addLayout(slider_layout)
        self.parameter_slider.valueChanged.connect(self.updateLineEdit)
        self.parameter_edit.textChanged.connect(self.updateSlider)

        second_slider_layout = QHBoxLayout()
        self.parameter_edit2 = QLineEdit()
        self.parameter_edit2.setFixedWidth(100)
        self.parameter_edit2.setText("50")
        self.parameter_edit2.setStyleSheet(
            "background-color: #337ab7; color: white; border-radius: 8px;"  # Style match
        )
        self.second_slider_label = QLabel("Frec de corte 2:")
        self.second_slider_label.setFont(font)
        self.second_parameter_slider = QSlider(Qt.Horizontal)
        self.second_parameter_slider.setMinimum(1)
        self.second_parameter_slider.setMaximum(int(self.sampFreq / 2))
        self.second_parameter_slider.setValue(50)
        self.second_parameter_slider.setTickInterval(100)
        second_slider_layout.addWidget(self.second_slider_label)
        second_slider_layout.addWidget(self.parameter_edit2)
        second_slider_layout.addWidget(self.second_parameter_slider)
        layout.addLayout(second_slider_layout)
        self.second_parameter_slider.valueChanged.connect(self.updateLineEdit2)
        self.parameter_edit2.textChanged.connect(self.updateSlider2)
        self.parameter_edit2.hide()
        self.second_slider_label.hide()
        self.second_parameter_slider.hide()

        self.apply_filter_button = QPushButton("Aplicar Filtro")
        self.apply_filter_button.setIconSize(QSize(40, 40))
        self.apply_filter_button.setStyleSheet(
            "background-color: #20aee6; border: 2px; color: #ffffff; border-radius: 30px;"
        )
        self.apply_filter_button.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.apply_filter_button.setFixedHeight(50)  # You can still set a fixed height

        # Add the button to the layout at the top
        layout.addWidget(self.apply_filter_button)

        # Add a stretch to push everything else down, making the button stay at the top
        layout.addStretch(1)

        self.apply_filter_button.clicked.connect(self.on_apply_filter_button_clicked)


        layout.addWidget(self.apply_filter_button, alignment=Qt.AlignCenter)
        self.filtro_tab.setLayout(layout)


    def on_apply_filter_button_clicked(self):
        """
        Handle the click event of the 'Apply Filter' button in the user interface.
        Retrieves the filter settings from the user interface elements and applies the filter based on these settings.
        """

        # Retrieve the selected filter type from the combo box (e.g., IIR, FIR)
        selected_filter_type = self.filter_type.currentText()

        # Retrieve the selected bandpass type from the combo box (e.g., lowpass, highpass, bandpass)
        selected_bandpass_type = self.bandpass.currentText()

        # Retrieve the primary cutoff frequency from the user input
        primary_freq_cutoff = float(self.parameter_edit.text())
        secondary_freq_cutoff = None  # Initialize secondary cutoff frequency to None

        # If the selected bandpass type is 'bandpass', retrieve the second cutoff frequency
        if selected_bandpass_type == "bandpass":
            secondary_freq_cutoff = float(self.parameter_edit2.text())

        # Retrieve the filter order from the spin box
        filter_order = self.order_spinbox.value()

        # Call the apply_filter method with the retrieved values to apply the specified filter
        self.apply_filter(selected_filter_type, selected_bandpass_type, primary_freq_cutoff, secondary_freq_cutoff, filter_order)


    def apply_filter(self, filter_type, band_type, primary_cutoff, secondary_cutoff=None, order=4, filter_design="butter"):
        """
        Apply the specified filter to the audio signal based on user inputs from the interface.
        This method coordinates the filtering process including setting up the filter parameters,
        applying the filter, and updating the interface to show the results.
        """

        # Determine the appropriate cutoff frequency format based on the filter type
        cutoff_frequency = self.get_cutoff_frequency(band_type, primary_cutoff, secondary_cutoff)

        # Apply the specified filter type with the determined parameters
        if filter_type == "IIR":
            self.apply_iir_filter(order, cutoff_frequency, band_type, filter_design)
        elif filter_type == "FIR":
            self.apply_fir_filter(order, cutoff_frequency, band_type)

        # After applying the filter, plot the filtered audio signal
        self.plot_filter()

        # Perform a Fourier Transform on the filtered signal to analyze its spectrum
        self.fourier_filter()

        # Update the interface to reflect changes and show the filtered signal and its Fourier spectrum
        self.update_filter_interface()


    def get_cutoff_frequency(self, band_type, f_cutoff1, f_cutoff2):
        """Calculate and return the appropriate cutoff frequency format."""
        if band_type == "bandpass" and f_cutoff2 is not None:
            return [f_cutoff1, f_cutoff2]  # For bandpass, we need a range of frequencies
        return f_cutoff1  # For other types, a single frequency is enough

    def apply_iir_filter(self, order, f_cutoff, band_type, ftype):
        """Apply an IIR filter with the given parameters."""
        b, a = iirfilter(N=order, Wn=f_cutoff, fs=self.sampFreq, btype=band_type, ftype=ftype)
        self.newSignal = filtfilt(b, a, self.sound)

    def apply_fir_filter(self, order, f_cutoff, band_type):
        """Apply an FIR filter with the given parameters."""
        width = 5.0 / (self.sampFreq / 2)  # Transition width of the filter
        ripple_db = 20.0  # Maximum ripple allowed (dB)
        N, beta = kaiserord(ripple_db, width)  # Calculate the order and beta for the Kaiser window

        # Ensure the order is odd for symmetry
        if N % 2 == 0:
            N += 1

        # Adjust cutoff frequency if needed
        if isinstance(f_cutoff, list):
            f_cutoff = [fc / (self.sampFreq / 2) for fc in f_cutoff]  # Normalize frequency for FIR
        else:
            f_cutoff = f_cutoff / (self.sampFreq / 2)

        # Create and apply the FIR filter using the Kaiser window
        taps = firwin(N, f_cutoff, window=("kaiser", beta), pass_zero=band_type, fs=self.sampFreq)
        self.newSignal = lfilter(taps, 1.0, self.sound)



    def fourier_filter(self):
        """Aplicar la ft y graficar la señal filtrada."""
        fft_spectrum = np.fft.rfft(self.newSignal)
        freq = np.fft.rfftfreq(self.newSignal.size, d=1.0 / self.sampFreq)
        fft_spectrum_abs = np.abs(fft_spectrum)

        plt.plot(freq, fft_spectrum_abs, "#337ab7")
        plt.xlabel("frequency, Hz")
        plt.ylabel("Amplitude, units")
        plt.title("Transformada de Fourier filtrada")
        plt.tight_layout()
        plt.savefig("Filtered_Fourier.png")
        plt.close()

    def plot_filter(self):
        """Graficar y guardar señal filtrada."""
        plt.plot(self.sound, label="Original Audio", color="#337ab7")
        plt.plot(self.newSignal, label="Filtered Audio", color="#0a0a0a")
        plt.title("Original vs. Filtered Audio Signal")
        plt.xlabel("Samples")
        plt.ylabel("Amplitude")
        plt.legend()
        plt.tight_layout()
        plt.savefig("Filtered_Audio.png")
        plt.close()

    def update_filter_interface(self):
        """Pestaña de Filtros se actualiza."""
        layout = self.filtro_tab.layout()

        # Mostrar graficas del audio filtrado
        self.img_layout = QHBoxLayout()
        self.img_layout.addWidget(self.show_image("Filtered_Audio.png", 1200, 600))
        self.img_layout.addWidget(self.show_image("Filtered_Fourier.png", 1200, 600))

        self.button_layout = QHBoxLayout()
        # Configure the 'Clear' button
        self.clear_button = QPushButton("Borrar")
        self.clear_button.setStyleSheet(
            "background-color: #337ab7; border: 2px; color: #ffffff; border-radius: 30px;"
        )
        self.clear_button.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)  # Allow horizontal expansion
        self.clear_button.setFixedHeight(50)  # Set a fixed height
        self.clear_button.clicked.connect(self.clear)  # Connect the 'clear' functionality
        self.button_layout.addWidget(self.clear_button)  # Add to the layout

        # Configure the 'Save' button
        self.save_button = QPushButton("Guardar")
        self.save_button.setStyleSheet(
            "background-color: #337ab7; border: 2px; color: #ffffff; border-radius: 30px;"
        )
        self.save_button.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)  # Allow horizontal expansion
        self.save_button.setFixedHeight(50)  # Set a fixed height
        self.button_layout.addWidget(self.save_button)  # Add to the layout
        self.filtro_tab.setLayout(self.button_layout)


        self.save_button.clicked.connect(self.save_filter)  # Guardar audio filtrado

        self.button_layout.addWidget(self.clear_button, alignment=Qt.AlignCenter)
        self.button_layout.addWidget(self.save_button, alignment=Qt.AlignCenter)

        layout.addLayout(self.button_layout)
        layout.insertLayout(6, self.img_layout)
        layout.insertLayout(7, self.button_layout)

    # ----------- CONFIGURACIONES -----------

    def clear(self):
        """Resetear valores."""
        self.newSignal = None  # borrar senal filtrada
        self.parameter_edit.setText("50")  # reset de slider de frec de corte 1
        self.parameter_edit2.setText("50")  # reset de slider de frec de corte 2
        self.order_spinbox.setValue(4)  # reset de orden

        if self.img_layout is not None:  # elimina graficas
            self.clearLayout(self.img_layout)
            self.img_layout = None

        if self.button_layout is not None:  # elimina botones
            self.clearLayout(self.button_layout)
            self.button_layout = None

    def clearLayout(self, layout):
        """Borrar elementos graficos."""
        if layout is not None:
            while layout.count():
                item = layout.takeAt(0)
                widget = item.widget()
                if widget is not None:
                    widget.deleteLater()
                else:
                    self.clearLayout(item.layout())

    def save_filter(self):
        """Guarda la señal filtrada en formato seleccionado por el usuario."""
        self.normalize_signal()
        audio_segment = self.create_audio_segment()
        self.save_audio_file(audio_segment)

    def normalize_signal(self):
        """Normaliza la señal de audio para ajustarse al rango de 16 bits."""
        max_amplitude = np.max(np.abs(self.newSignal))
        self.newSignal = np.int16(self.newSignal / max_amplitude * 32767)

    def create_audio_segment(self):
        """Crea un segmento de audio a partir de la señal normalizada."""
        return AudioSegment(
            self.newSignal.tobytes(),
            sample_width=2,
            frame_rate=self.sampFreq,
            channels=1
        )

    def save_audio_file(self, audio_segment):
        """Muestra un diálogo para guardar el archivo y lo guarda en el formato seleccionado."""
        options = QFileDialog.Options()
        fileName, selectedFilter = QFileDialog.getSaveFileName(
            self,
            "Guardar archivo",
            "",
            "WAV Audio Files (*.wav);;MP3 Audio Files (*.mp3);;AAC Audio Files (*.aac)",
            options=options
        )

        if fileName:
            self.append_file_extension(fileName, selectedFilter)
            self.export_audio_file(fileName, audio_segment, selectedFilter)

    def append_file_extension(self, fileName, selectedFilter):
        """Asegura que el nombre del archivo tenga la extensión correcta."""
        if ".wav" in selectedFilter and not fileName.endswith(".wav"):
            return fileName + ".wav"
        elif ".mp3" in selectedFilter and not fileName.endswith(".mp3"):
            return fileName + ".mp3"
        elif ".aac" in selectedFilter and not fileName.endswith(".aac"):
            return fileName + ".aac"
        return fileName

    def export_audio_file(self, fileName, audio_segment, selectedFilter):
        """Exporta el archivo de audio en el formato especificado."""
        if fileName.endswith(".wav"):
            wavfile.write(fileName, self.sampFreq, self.newSignal)
        elif fileName.endswith(".mp3"):
            audio_segment.export(fileName, format="mp3")
        elif fileName.endswith(".aac"):
            audio_segment.export(fileName, format="ipod", codec="aac")


    def updateSlider(self):
        """
        Update the slider's value based on the text input in the corresponding line edit.
        """
        value = int(self.parameter_edit.text())  # Convert the text from the line edit to an integer
        self.parameter_slider.setValue(value)  # Set the slider's value to the converted integer

    def updateLineEdit(self):
        """
        Update the line edit's text based on the value of the corresponding slider.
        """
        value = self.parameter_slider.value()  # Get the current value of the slider
        self.parameter_edit.setText(str(value))  # Set the line edit's text to the slider's value

    def updateSlider2(self):
        """
        Update the second slider's value based on the text input in the corresponding second line edit.
        """
        value = int(self.parameter_edit2.text())  # Convert the text from the second line edit to an integer
        self.second_parameter_slider.setValue(value)  # Set the second slider's value to the converted integer

    def updateLineEdit2(self):
        """
        Update the second line edit's text based on the value of the corresponding second slider.
        """
        value = self.second_parameter_slider.value()  # Get the current value of the second slider
        self.parameter_edit2.setText(str(value))  # Set the second line edit's text to the second slider's value

    def updateBandpass(self, index):
        """
        Toggle visibility of the second cutoff frequency controls based on the selected filter type.
        Shows or hides elements related to the second cutoff frequency when 'bandpass' is selected.
        """
        if index == 2:  # If 'bandpass' is selected
            self.parameter_edit2.show()  # Show the second line edit
            self.second_slider_label.show()  # Show the label for the second slider
            self.second_parameter_slider.show()  # Show the second slider
        else:  # For other filter types ('lowpass', 'highpass')
            self.parameter_edit2.hide()  # Hide the second line edit
            self.second_slider_label.hide()  # Hide the label for the second slider
            self.second_parameter_slider.hide()  # Hide the second slider

    def show_image(self, image_path, width, height):
        """
        Display an image in the interface from a specified path, resized to given dimensions.
        """
        pixmap = QPixmap(image_path)  # Load the image from the file path
        pixmap = pixmap.scaled(width, height, Qt.KeepAspectRatio)  # Scale the pixmap to the specified width and height while maintaining aspect ratio
        label = QLabel(self)  # Create a QLabel to hold the pixmap
        label.setPixmap(pixmap)  # Set the pixmap into the QLabel
        label.setAlignment(Qt.AlignCenter)  # Align the label to be centered
        return label  # Return the label with the image



def main():
    app = QApplication([])
    window = MyWindow()
    window.show()
    app.exec_()


if __name__ == "__main__":
    main()