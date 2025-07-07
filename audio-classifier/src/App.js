import React, { useState, useRef, useEffect } from 'react';
import { Upload, Play, Loader2, XCircle, CheckCircle, Brain, Mic2, Waves, LayoutGrid, Layers } from 'lucide-react';

// Helper function to draw a 2D array onto a canvas (e.g., spectrogram, feature map)
const draw2DArray = (canvas, data, title = "") => {
    if (!canvas || !data || data.length === 0 || data[0].length === 0) {
        return;
    }
    const ctx = canvas.getContext('2d');
    const height = data.length;
    const width = data[0].length;

    // Set canvas dimensions to match data for pixel-perfect rendering
    canvas.width = width;
    canvas.height = height;

    const imageData = ctx.createImageData(width, height);
    const minVal = Math.min(...data.flat());
    const maxVal = Math.max(...data.flat());

    for (let y = 0; y < height; y++) {
        for (let x = 0; x < width; x++) {
            const value = data[y][x];
            // Normalize value to 0-255 range for grayscale
            const normalized = maxVal === minVal ? 0 : Math.floor(((value - minVal) / (maxVal - minVal)) * 255);
            const index = (y * width + x) * 4;
            imageData.data[index] = normalized;     // Red
            imageData.data[index + 1] = normalized; // Green
            imageData.data[index + 2] = normalized; // Blue
            imageData.data[index + 3] = 255;        // Alpha
        }
    }
    ctx.putImageData(imageData, 0, 0);

    // Optional: Add title for context
    if (title) {
        ctx.font = '12px Inter';
        ctx.fillStyle = 'rgba(255, 255, 255, 0.8)'; // Semi-transparent white
        ctx.fillText(title, 5, 15);
    }
};

// Helper function to draw a waveform onto a canvas
const drawWaveform = (canvas, data) => {
    if (!canvas || !data || data.length === 0) {
        return;
    }
    const ctx = canvas.getContext('2d');
    const width = canvas.width;
    const height = canvas.height;
    ctx.clearRect(0, 0, width, height); // Clear previous drawings
    ctx.strokeStyle = '#6366f1'; // Indigo 500
    ctx.lineWidth = 1.5;

    const maxVal = Math.max(...data.map(Math.abs));
    const scale = (height / 2) / (maxVal || 1); // Avoid division by zero

    ctx.beginPath();
    ctx.moveTo(0, height / 2); // Start at the middle line

    for (let i = 0; i < data.length; i++) {
        const x = (i / data.length) * width;
        const y = height / 2 - data[i] * scale;
        ctx.lineTo(x, y);
    }
    ctx.stroke();
};

// --- Mock AudioClassifier (Simulates your Python backend) ---
class MockAudioClassifier {
    constructor() {
        this.classes = ["Speech", "Music", "Noise", "Vehicle", "Animal", "Nature", "Human", "Mechanical", "Alarm", "Water"]; // More example classes
        this.mockFeatureMapNames = ['Input Conv', 'Pool 1', 'Conv 2', 'Pool 2', 'Global Pool']; // More descriptive names
        console.log("Mock AudioClassifier initialized.");
    }

    load_model() {
        // In a real app, this would make an API call to load the model.
        console.log("Simulating model loading...");
        return Promise.resolve();
    }

    async inference(audio_b64_string) {
        console.log("Simulating inference...");
        // Simulate network delay
        await new Promise(resolve => setTimeout(2500)); // Slightly longer delay

        let audioBytesLength = 0;
        try {
            audioBytesLength = atob(audio_b64_string).length;
        } catch (e) {
            console.error("Invalid base64 string for mock inference:", e);
            audioBytesLength = 10000; // Default if decoding fails
        }

        // Generate mock predictions
        const mockPredictions = [];
        const numClasses = this.classes.length;
        const randomConfidences = Array.from({ length: numClasses }, () => Math.random());
        const sumConfidences = randomConfidences.reduce((a, b) => a + b, 0);
        const normalizedConfidences = randomConfidences.map(c => c / sumConfidences);

        const sortedPredictions = this.classes.map((className, index) => ({
            class: className,
            confidence: normalizedConfidences[index]
        })).sort((a, b) => b.confidence - a.confidence);

        for (let i = 0; i < Math.min(5, numClasses); i++) { // Always top 5
            mockPredictions.push(sortedPredictions[i]);
        }

        // Generate mock input spectrogram (e.g., 128 Mel bins, variable time frames)
        const mockSpectrogramHeight = 128;
        const mockSpectrogramWidth = Math.max(80, Math.min(400, Math.floor(audioBytesLength / 80))); // Scale width based on audio size
        const mockInputSpectrogram = Array.from({ length: mockSpectrogramHeight }, () =>
            Array.from({ length: mockSpectrogramWidth }, () => Math.random() * 15 - 7.5) // More varied random values
        );

        // Generate mock waveform
        const mockWaveformLength = Math.min(12000, Math.floor(audioBytesLength / 8)); // Max 12000 samples
        const mockWaveform = Array.from({ length: mockWaveformLength }, (_, i) => Math.sin(i * 0.05) * 0.7 + (Math.random() - 0.5) * 0.3);

        // Generate mock feature maps - simulating reduction in size and increase in complexity
        const mockVizData = {};
        let currentHeight = mockSpectrogramHeight;
        let currentWidth = mockSpectrogramWidth;

        this.mockFeatureMapNames.forEach((name, index) => {
            // Simulate pooling/convolution reducing dimensions
            if (index > 0) { // Simulate reduction after the first layer
                currentHeight = Math.max(8, Math.floor(currentHeight / (index === 0 ? 1 : 2))); // Halve height
                currentWidth = Math.max(8, Math.floor(currentWidth / (index === 0 ? 1 : 2)));   // Halve width
            }
            // For 'Global Pool', reduce to 1x1
            if (name === 'Global Pool') {
                currentHeight = 1;
                currentWidth = 1;
            }

            const mapHeight = currentHeight;
            const mapWidth = currentWidth;

            mockVizData[name] = {
                shape: [mapHeight, mapWidth],
                values: Array.from({ length: mapHeight }, () =>
                    Array.from({ length: mapWidth }, () => Math.random() * 8 - 4) // Different range for feature maps
                )
            };
        });

        return {
            predictions: mockPredictions,
            visualization: mockVizData,
            input_spectrogram: {
                shape: [mockSpectrogramHeight, mockSpectrogramWidth],
                values: mockInputSpectrogram
            },
            waveform: {
                values: mockWaveform,
                sample_rate: 44100,
                duration: mockWaveformLength / 44100
            }
        };
    }
}

const mockClassifier = new MockAudioClassifier();

function App() {
    const [selectedFile, setSelectedFile] = useState(null);
    const [predictionResult, setPredictionResult] = useState(null);
    const [loading, setLoading] = useState(false);
    const [error, setError] = useState(null);
    const [audioUrl, setAudioUrl] = useState(null);

    const spectrogramCanvasRef = useRef(null);
    const waveformCanvasRef = useRef(null);
    const featureMapCanvasRefs = useRef({});

    // Effect for drawing spectrogram
    useEffect(() => {
        if (predictionResult && spectrogramCanvasRef.current) {
            draw2DArray(spectrogramCanvasRef.current, predictionResult.input_spectrogram.values, "Input Mel-Spectrogram");
        }
    }, [predictionResult?.input_spectrogram]);

    // Effect for drawing waveform
    useEffect(() => {
        if (predictionResult && waveformCanvasRef.current) {
            drawWaveform(waveformCanvasRef.current, predictionResult.waveform.values);
        }
    }, [predictionResult?.waveform]);

    // Effect for drawing feature maps
    useEffect(() => {
        if (predictionResult && predictionResult.visualization) {
            Object.entries(predictionResult.visualization).forEach(([name, data]) => {
                const canvas = featureMapCanvasRefs.current[name];
                if (canvas) {
                    draw2DArray(canvas, data.values, name);
                }
            });
        }
    }, [predictionResult?.visualization]);

    const handleFileChange = (event) => {
        const file = event.target.files[0];
        if (file) {
            setSelectedFile(file);
            setError(null);
            setPredictionResult(null);
            setAudioUrl(URL.createObjectURL(file));
        }
    };

    const handlePredict = async () => {
        if (!selectedFile) {
            setError("Please upload an audio file first.");
            return;
        }

        setLoading(true);
        setError(null);
        setPredictionResult(null);

        try {
            const reader = new FileReader();
            reader.readAsArrayBuffer(selectedFile);
            reader.onload = async (e) => {
                const audioBytes = new Uint8Array(e.target.result);
                const audio_b64 = btoa(String.fromCharCode.apply(null, audioBytes));

                await mockClassifier.load_model(); // Simulate model loading
                const result = await mockClassifier.inference(audio_b64);
                setPredictionResult(result);
            };
            reader.onerror = (e) => {
                setError("Failed to read audio file.");
                console.error("FileReader error:", e);
                setLoading(false);
            };

        } catch (err) {
            console.error("Prediction error:", err);
            setError("Prediction failed. " + err.message);
        } finally {
            setLoading(false);
        }
    };

    return (
        <div className="min-h-screen bg-gradient-to-br from-gray-950 to-black p-4 sm:p-8 font-inter text-gray-100">
            <div className="max-w-7xl mx-auto bg-neutral-900 rounded-2xl shadow-xl overflow-hidden border border-neutral-800">
                {/* Header */}
                <header className="bg-neutral-900 text-white p-6 sm:p-8 flex items-center justify-center rounded-t-2xl border-b border-neutral-800">
                    <h1 className="text-2xl sm:text-4xl font-bold flex items-center gap-4 text-white">
                        <Brain className="w-8 h-8 sm:w-10 sm:h-10 text-indigo-400" />
                        Audio Classifier using CNN
                    </h1>
                </header>

                <main className="p-6 sm:p-8 flex flex-col gap-8">
                    {/* Upload and Predict Section */}
                    <section className="w-full bg-neutral-800 p-8 rounded-xl shadow-inner border border-neutral-700 flex flex-col items-center">
                        <h2 className="text-xl font-semibold mb-6 text-indigo-400 flex items-center gap-2">
                            <Upload className="w-5 h-5" /> Upload Audio
                        </h2>
                        <div className="mb-6 w-full max-w-md"> {/* Max width for input for cleaner look */}
                            <label htmlFor="audio-upload" className="block text-sm font-medium text-gray-300 mb-2">
                                Select an audio file (.wav, .mp3, etc.)
                            </label>
                            <input
                                type="file"
                                id="audio-upload"
                                accept="audio/*"
                                onChange={handleFileChange}
                                className="block w-full text-sm text-gray-100
                                           file:mr-4 file:py-2 file:px-4
                                           file:rounded-full file:border-0
                                           file:text-sm file:font-semibold
                                           file:bg-indigo-700 file:text-white
                                           hover:file:bg-indigo-600 cursor-pointer
                                           focus:outline-none focus:ring-2 focus:ring-indigo-500 focus:ring-offset-2 focus:ring-offset-neutral-800"
                            />
                            {selectedFile && (
                                <p className="mt-3 text-sm text-gray-300">
                                    Selected: <span className="font-medium text-white">{selectedFile.name}</span>
                                </p>
                            )}
                            {audioUrl && (
                                <div className="mt-4 flex items-center gap-3 p-2 bg-neutral-700 rounded-lg shadow-md">
                                    <Play className="text-indigo-400 w-6 h-6" />
                                    <audio controls src={audioUrl} className="w-full"></audio>
                                </div>
                            )}
                        </div>

                        <button
                            onClick={handlePredict}
                            disabled={!selectedFile || loading}
                            className="w-full max-w-md flex items-center justify-center gap-3 px-6 py-3 border border-transparent text-base font-medium rounded-lg shadow-lg text-white bg-indigo-600 hover:bg-indigo-700 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-indigo-500 focus:ring-offset-neutral-800 transition-all duration-200 ease-in-out disabled:opacity-40 disabled:cursor-not-allowed transform hover:scale-105"
                        >
                            {loading ? (
                                <Loader2 className="animate-spin h-6 w-6" />
                            ) : (
                                <Brain className="h-6 w-6" />
                            )}
                            {loading ? 'Analyzing...' : 'Predict Sound'}
                        </button>

                        {error && (
                            <div className="mt-6 p-4 bg-red-800 border border-red-600 text-red-200 rounded-lg flex items-center gap-3 shadow-md w-full max-w-md">
                                <XCircle className="h-6 w-6" />
                                <p className="text-sm font-medium">{error}</p>
                            </div>
                        )}
                        {predictionResult && !loading && (
                            <div className="mt-6 p-4 bg-green-800 border border-green-600 text-green-200 rounded-lg flex items-center gap-3 shadow-md w-full max-w-md">
                                <CheckCircle className="h-6 w-6" />
                                <p className="text-sm font-medium">Prediction complete!</p>
                            </div>
                        )}
                    </section>

                    {/* Results Display Section */}
                    <section className="w-full bg-neutral-800 p-8 rounded-xl shadow-inner border border-neutral-700">
                        <h2 className="text-xl font-semibold mb-8 text-purple-400 flex items-center gap-2">
                            <LayoutGrid className="w-5 h-5" /> Prediction Results
                        </h2>

                        {predictionResult ? (
                            <div className="space-y-10"> {/* Increased vertical spacing between sections */}
                                {/* Top Predictions */}
                                <div className="bg-neutral-900 p-6 rounded-lg shadow-md border border-neutral-700">
                                    <h3 className="text-lg font-medium mb-4 text-white flex items-center gap-2">
                                        <Mic2 className="w-4 h-4 text-indigo-400" /> Top 5 Predictions
                                    </h3>
                                    <ul className="space-y-3"> {/* Increased spacing between list items */}
                                        {predictionResult.predictions.map((pred, index) => (
                                            <li key={index} className="flex items-center justify-between text-base text-gray-200">
                                                <span className="font-semibold">{pred.class}</span>
                                                <span className="text-indigo-400 font-bold text-lg">
                                                    {(pred.confidence * 100).toFixed(2)}%
                                                </span>
                                            </li>
                                        ))}
                                    </ul>
                                </div>

                                {/* Input Spectrogram and Audio Waveform */}
                                <div className="grid grid-cols-1 lg:grid-cols-2 gap-6"> {/* Use lg:grid-cols-2 for larger screens */}
                                    {/* Waveform Visualization */}
                                    <div className="bg-neutral-900 p-4 rounded-lg shadow-md border border-neutral-700">
                                        <h3 className="text-lg font-medium mb-3 text-white flex items-center gap-2">
                                            <Waves className="w-4 h-4 text-purple-400" /> Input Waveform
                                        </h3>
                                        <canvas ref={waveformCanvasRef} className="w-full h-32 rounded-md bg-neutral-800 border border-neutral-600"></canvas>
                                        <p className="text-xs text-gray-400 mt-2 text-center">
                                            Duration: {predictionResult.waveform.duration.toFixed(2)}s,
                                            Sample Rate: {predictionResult.waveform.sample_rate} Hz
                                        </p>
                                    </div>

                                    {/* Input Spectrogram */}
                                    <div className="bg-neutral-900 p-4 rounded-lg shadow-md border border-neutral-700">
                                        <h3 className="text-lg font-medium mb-3 text-white flex items-center gap-2">
                                            <LayoutGrid className="w-4 h-4 text-purple-400" /> Input Mel-Spectrogram
                                        </h3>
                                        <canvas ref={spectrogramCanvasRef} className="w-full h-48 rounded-md bg-neutral-800 border border-neutral-600"></canvas>
                                        <p className="text-xs text-gray-400 mt-2 text-center">
                                            Shape: {predictionResult.input_spectrogram.shape[0]} (Mel Bins) x {predictionResult.input_spectrogram.shape[1]} (Time Frames)
                                        </p>
                                    </div>
                                </div>

                                {/* Feature Map Visualizations (Convolution Layers) */}
                                <div className="bg-neutral-900 p-6 rounded-lg shadow-md border border-neutral-700">
                                    <h3 className="text-lg font-medium mb-6 text-white flex items-center gap-2">
                                        <Layers className="w-4 h-4 text-green-400" /> Convolution Steps (Feature Maps)
                                    </h3>
                                    <div className="grid grid-cols-1 sm:grid-cols-2 md:grid-cols-3 lg:grid-cols-4 xl:grid-cols-5 gap-4">
                                        {mockClassifier.mockFeatureMapNames.map((name) => (
                                            <div key={name} className="bg-neutral-800 rounded-md p-3 shadow-inner border border-neutral-700 flex flex-col items-center">
                                                <h4 className="text-sm font-medium mb-2 text-center text-gray-200">{name}</h4>
                                                <canvas
                                                    ref={el => featureMapCanvasRefs.current[name] = el}
                                                    className="w-full h-24 rounded-md bg-neutral-900 border border-neutral-600"
                                                ></canvas>
                                                {predictionResult.visualization[name] && (
                                                    <p className="text-xs text-gray-400 mt-2 text-center">
                                                        Shape: {predictionResult.visualization[name].shape[0]}x{predictionResult.visualization[name].shape[1]}
                                                    </p>
                                                )}
                                            </div>
                                        ))}
                                    </div>
                                </div>
                            </div>
                        ) : (
                            <p className="text-gray-400 text-center py-10 text-lg">
                                Upload an audio file and click "Predict Sound" to see the magic unfold.
                            </p>
                        )}
                    </section>
                </main>

                {/* Footer */}
                <footer className="bg-neutral-900 text-gray-400 text-center p-4 rounded-b-2xl text-sm border-t border-neutral-800">
                    Crafted with <span className="text-red-500">♥</span> using React & Simulated AI.
                </footer>
            </div>
        </div>
    );
}

export default App;
