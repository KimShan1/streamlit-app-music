

import os 
import streamlit as st
import pandas as pd
import joblib
import essentia.standard
from essentia.streaming import *
import numpy as np
import essentia.standard
import numpy as np
from contextlib import redirect_stdout, redirect_stderr
from essentia.streaming import (
    FrameCutter, Windowing, Spectrum, MFCC, ZeroCrossingRate, Centroid, Energy, RMS, Decrease, CentralMoments,
    DistributionShape, LPC, BarkBands, FlatnessDB, Crest, SpectralContrast, SpectralPeaks, HarmonicPeaks, 
    Dissonance, Tristimulus, OddToEvenHarmonicEnergyRatio, Inharmonicity, RollOff, HFC, Flux, SpectralComplexity, 
    PitchYinFFT, PitchSalience, EqloudLoader, UnaryOperator, EnergyBand, StrongPeak, CompositeBase
)
import io
from essentia import Pool, run
import logging
from streamlit.components.v1 import html

# Set logging level to suppress debug information
logging.getLogger().setLevel(logging.WARNING)

# Load the model, scaler, label encoder, and feature names
best_model = joblib.load('best_xgboost_model.pkl')
scaler = joblib.load('scaler.pkl')
label_encoder = joblib.load('label_encoder.pkl')
feature_names = joblib.load('feature_names.pkl')

FILE_EXT = "*.wav"

class FeatureExtractor(essentia.streaming.CompositeBase):

    def __init__(self, frameSize=2048, hopSize=1024, sampleRate=44100.):
        super(FeatureExtractor, self).__init__()

        halfSampleRate = sampleRate / 2
        minFrequency = sampleRate / frameSize

        fc = FrameCutter(frameSize=frameSize, hopSize=hopSize)
        zcr = ZeroCrossingRate()
        fc.frame >> zcr.signal
        w = Windowing(type='blackmanharris62')
        fc.frame >> w.frame
        spec = Spectrum()
        w.frame >> spec.frame
        energy = Energy()
        spec.spectrum >> energy.array
        rms = RMS()
        spec.spectrum >> rms.array
        square1 = UnaryOperator(type='square')
        centroid = Centroid(range=halfSampleRate)
        spec.spectrum >> square1.array >> centroid.array
        cm = CentralMoments(range=halfSampleRate)
        ds = DistributionShape()
        spec.spectrum >> cm.array
        cm.centralMoments >> ds.centralMoments
        mfcc = MFCC(numberBands=40, numberCoefficients=13, sampleRate=sampleRate)
        spec.spectrum >> mfcc.spectrum
        mfcc.bands >> None
        lpc = LPC(order=10, sampleRate=sampleRate)
        spec.spectrum >> lpc.frame
        lpc.reflection >> None
        square2 = UnaryOperator(type='square')
        decrease = Decrease(range=halfSampleRate)
        spec.spectrum >> square2.array >> decrease.array
        ebr_low = EnergyBand(startCutoffFrequency=20, stopCutoffFrequency=150, sampleRate=sampleRate)
        ebr_mid_low = EnergyBand(startCutoffFrequency=150, stopCutoffFrequency=800, sampleRate=sampleRate)
        ebr_mid_hi = EnergyBand(startCutoffFrequency=800, stopCutoffFrequency=4000, sampleRate=sampleRate)
        ebr_hi = EnergyBand(startCutoffFrequency=4000, stopCutoffFrequency=20000, sampleRate=sampleRate)
        spec.spectrum >> ebr_low.spectrum
        spec.spectrum >> ebr_mid_low.spectrum
        spec.spectrum >> ebr_mid_hi.spectrum
        spec.spectrum >> ebr_hi.spectrum
        hfc = HFC(sampleRate=sampleRate)
        spec.spectrum >> hfc.spectrum
        flux = Flux()
        spec.spectrum >> flux.spectrum
        ro = RollOff(sampleRate=sampleRate)
        spec.spectrum >> ro.spectrum
        sp = StrongPeak()
        spec.spectrum >> sp.spectrum
        barkBands = BarkBands(numberBands=27, sampleRate=sampleRate)
        spec.spectrum >> barkBands.spectrum
        crest = Crest()
        barkBands.bands >> crest.array
        flatness = FlatnessDB()
        barkBands.bands >> flatness.array
        cmbb = CentralMoments(range=26)
        dsbb = DistributionShape()
        barkBands.bands >> cmbb.array
        cmbb.centralMoments >> dsbb.centralMoments
        scx = SpectralComplexity(magnitudeThreshold=0.005, sampleRate=sampleRate)
        spec.spectrum >> scx.spectrum
        pitch = PitchYinFFT(frameSize=frameSize, sampleRate=sampleRate)
        spec.spectrum >> pitch.spectrum
        pitch.pitch >> None
        ps = PitchSalience(sampleRate=sampleRate)
        spec.spectrum >> ps.spectrum
        sc = SpectralContrast(frameSize=frameSize, sampleRate=sampleRate, numberBands=6, lowFrequencyBound=20, highFrequencyBound=11000, neighbourRatio=0.4, staticDistribution=0.15)
        spec.spectrum >> sc.spectrum
        peaks = SpectralPeaks(orderBy='frequency', minFrequency=minFrequency, sampleRate=sampleRate)
        spec.spectrum >> peaks.spectrum
        diss = Dissonance()
        peaks.frequencies >> diss.frequencies
        peaks.magnitudes >> diss.magnitudes
        harmPeaks = HarmonicPeaks()
        peaks.frequencies >> harmPeaks.frequencies
        peaks.magnitudes >> harmPeaks.magnitudes
        pitch.pitch >> harmPeaks.pitch
        tristimulus = Tristimulus()
        harmPeaks.harmonicFrequencies >> tristimulus.frequencies
        harmPeaks.harmonicMagnitudes >> tristimulus.magnitudes
        odd2even = OddToEvenHarmonicEnergyRatio()
        harmPeaks.harmonicFrequencies >> odd2even.frequencies
        harmPeaks.harmonicMagnitudes >> odd2even.magnitudes
        inharmonicity = Inharmonicity()
        harmPeaks.harmonicFrequencies >> inharmonicity.frequencies
        harmPeaks.harmonicMagnitudes >> inharmonicity.magnitudes

        self.inputs['signal'] = fc.signal
        self.outputs['zcr'] = zcr.zeroCrossingRate
        self.outputs['spectral_energy'] = energy.energy
        self.outputs['spectral_rms'] = rms.rms
        self.outputs['mfcc'] = mfcc.mfcc
        self.outputs['lpc'] = lpc.lpc
        self.outputs['spectral_centroid'] = centroid.centroid
        self.outputs['spectral_kurtosis'] = ds.kurtosis
        self.outputs['spectral_spread'] = ds.spread
        self.outputs['spectral_skewness'] = ds.skewness
        self.outputs['spectral_dissonance'] = diss.dissonance
        self.outputs['sccoeffs'] = sc.spectralContrast
        self.outputs['scvalleys'] = sc.spectralValley
        self.outputs['spectral_decrease'] = decrease.decrease
        self.outputs['spectral_energyband_low'] = ebr_low.energyBand
        self.outputs['spectral_energyband_middle_low'] = ebr_mid_low.energyBand
        self.outputs['spectral_energyband_middle_high'] = ebr_mid_hi.energyBand
        self.outputs['spectral_energyband_high'] = ebr_hi.energyBand
        self.outputs['hfc'] = hfc.hfc
        self.outputs['spectral_flux'] = flux.flux
        self.outputs['spectral_rolloff'] = ro.rollOff
        self.outputs['spectral_strongpeak'] = sp.strongPeak
        self.outputs['barkbands'] = barkBands.bands
        self.outputs['spectral_crest'] = crest.crest
        self.outputs['spectral_flatness_db'] = flatness.flatnessDB
        self.outputs['barkbands_kurtosis'] = dsbb.kurtosis
        self.outputs['barkbands_spread'] = dsbb.spread
        self.outputs['barkbands_skewness'] = dsbb.skewness
        self.outputs['spectral_complexity'] = scx.spectralComplexity
        self.outputs['pitch_instantaneous_confidence'] = pitch.pitchConfidence
        self.outputs['pitch_salience'] = ps.pitchSalience
        self.outputs['inharmonicity'] = inharmonicity.inharmonicity
        self.outputs['oddtoevenharmonicenergyratio'] = odd2even.oddToEvenHarmonicEnergyRatio
        self.outputs['tristimulus'] = tristimulus.tristimulus


def preprocess_song(file_path):
    loader = EqloudLoader(filename=file_path)
    fEx = FeatureExtractor(frameSize=2048, hopSize=1024, sampleRate=loader.paramValue('sampleRate'))
    p = Pool()

    loader.audio >> fEx.signal

    # for desc, output in fEx.outputs.items():
    # #     output >> (p, desc)
    # essentia.run(loader)

    # Suppress stdout and stderr
    f = io.StringIO()
    with redirect_stdout(f), redirect_stderr(f):
        for desc, output in fEx.outputs.items():
            output >> (p, desc)
        run(loader)

    stats = ['mean', 'var', 'dmean', 'dvar']
    statsPool = essentia.standard.PoolAggregator(defaultStats=stats)(p)
    
    pool_dict = dict()
    for desc in statsPool.descriptorNames():
        if type(statsPool[desc]) is float:
            pool_dict[desc] = statsPool[desc]
        elif type(statsPool[desc]) is np.ndarray:
            for i, value in enumerate(statsPool[desc]):
                feature_name = "{desc_name}{desc_number}.{desc_stat}".format(
                    desc_name=desc.split('.')[0],
                    desc_number=i,
                    desc_stat=desc.split('.')[1])
                pool_dict[feature_name] = value
    
    features = pd.DataFrame(pool_dict, index=[os.path.basename(file_path)])
    features = scaler.transform(features)
    return features

# Function to predict instruments in a song
def predict_instruments(model, file_path, scaler, label_encoder,feature_names):
    # Preprocess the song to extract features
    features = preprocess_song(file_path)

    # Ensure features are a DataFrame
    if not isinstance(features, pd.DataFrame):
        features = pd.DataFrame(features, columns=feature_names)
        
    # Ensure features have the same columns as those used during training
    features = features.reindex(columns=feature_names, fill_value=0)
    
    # Standardize the features
    features = scaler.transform(features)
    
    # Predict the probabilities of each class
    probabilities = model.predict_proba(features)[0]

    # Get the class labels
    classes = label_encoder.classes_
    
    # Create a sorted list of (class, probability) tuples
    sorted_probabilities = sorted(zip(classes, probabilities), key=lambda x: x[1], reverse=True)
    
    return sorted_probabilities
#####
def process_audio_file(uploaded_file, model, scaler, label_encoder, feature_names):
    with open("temp.wav", "wb") as f:
        f.write(uploaded_file.getbuffer())
    predicted_instruments = predict_instruments(best_model, "temp.wav", scaler, label_encoder, feature_names)
    return predicted_instruments
####

# Streamlit app
st.set_page_config(
    page_title = "Instrument Identification in Songs", 
    page_icon="🎶", 
    layout = "centered", 
    initial_sidebar_state = "expanded"
)
st.title('Instrument Identification in Songs')


uploaded_file = st.file_uploader("Choose an audio file...", type=["wav"])

if uploaded_file is not None:
    st.audio(uploaded_file, format='audio/wav')
    st.write("Processing the file...")
    predicted_instruments = process_audio_file(uploaded_file, best_model, scaler, label_encoder, feature_names)
    
    # Placeholder for the results
    result_placeholder = st.empty()
    # Display the results
    with result_placeholder.container():
        st.write("Predicted instruments (from highest to lowest probability):")
        for instrument, probability in predicted_instruments:
            st.write(f'{instrument}: {probability:.4f}')        

# JavaScript to scroll down to the results
    scroll_script = """
    <script>
    document.getElementById('result_placeholder').scrollIntoView({ behavior: 'smooth' });
    </script>
    """
    st.markdown(f"<div id='result_placeholder'></div>", unsafe_allow_html=True)
    st.markdown(scroll_script, unsafe_allow_html=True)
    html(scroll_script, height=100, scrolling=True)

# streamlit run music_app_predictor.py
