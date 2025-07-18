"use client"; // Required for Next.js 13+ App Router for client-side functionality

import './App.css'; // Your global or component-specific CSS
import React, { useState } from 'react'; // Import React and the useState hook
// Import Recharts components for the bar chart
import { BarChart, Bar, XAxis, YAxis, Tooltip, Legend, ResponsiveContainer, CartesianGrid } from 'recharts';

export default function App() {
  // State variables to manage UI and data flow
  const [isLoading, setIsLoading] = useState(false); // Tracks if an operation is in progress
  const [fileName, setFileName] = useState(null);   // Stores the name of the selected file
  const [error, setError] = useState(null);         // Stores any error messages
  const [vizData, setVizData] = useState(null);     // Stores the response data from the backend

  /**
   * Handles the file selection event and orchestrates the backend communication.
   * @param {object} event The change event from the file input.
   */
  const handleFileChange = async (event) => {
    const file = event.target.files?.[0]; // Get the first selected file

    // If no file was selected (e.g., user cancelled the dialog), clear states and exit
    if (!file) {
      setFileName(null);
      setError(null);
      setVizData(null); // Clear previous viz data
      return;
    }

    // 1. Update UI to reflect file selection and start loading
    setFileName(file.name); // Display the selected file's name
    setIsLoading(true);     // Activate loading indicator
    setError(null);         // Clear any old errors
    setVizData(null);       // Clear any old visualization data

    try {
      // 2. Prepare the file for sending to the backend using FormData
      const formData = new FormData();
      formData.append('audioFile', file); // 'audioFile' should match the field name your backend expects

      // 3. Send the POST request to your backend API
      //    The URL 'http://127.0.0.1:5000/api/predict' should match your FastAPI endpoint
      const response = await fetch('http://127.0.0.1:5000/api/predict', {
        method: 'POST',
        body: formData,
        // No 'Content-Type' header needed for FormData; browser handles it
      });

      // 4. Check if the backend response was successful (HTTP status 200-299)
      if (!response.ok) {
        let errorMessage = `HTTP error! Status: ${response.status}`;
        try {
          // Attempt to read a more specific error message from the backend's response body
          const errorBody = await response.json();
          if (errorBody.message) {
            errorMessage = errorBody.message;
          } else if (errorBody.error) {
            errorMessage = errorBody.error;
          }
        } catch (parseError) {
          // If response body isn't JSON, use generic message
          console.warn("Could not parse error response as JSON:", parseError);
        }
        throw new Error(errorMessage); // Throw an error to be caught by the catch block
      }

      // 5. Parse the successful JSON response from the backend
      const data = await response.json();
      console.log('Backend response:', data); // Log the full response to console for debugging

      // 6. Store the received data in state to trigger rendering of results
      setVizData(data);

    } catch (err) {
      // 7. Handle any errors that occurred during the fetch or processing
      console.error('Error during file processing:', err);
      setError(err.message || 'An unexpected error occurred during processing.'); // Display error message to user
    } finally {
      // 8. Always set loading state to false, regardless of success or failure
      setIsLoading(false);
    }
  };

  // Prepare data for the chart: Get top 3 predictions and format them
  // This logic runs on every render, but only processes `vizData.predictions` when available
  const chartData = vizData && vizData.predictions
    ? vizData.predictions
        .sort((a, b) => b.confidence - a.confidence) // Sort by confidence descending
        .slice(0, 3) // Take the top 3
        .map(pred => ({
          name: pred.class.replace(/_/g, ' '), // Format class name for display (e.g., "washing_machine" -> "washing machine")
          Probability: (pred.confidence * 100).toFixed(2), // Convert to percentage and format to 2 decimal places
        }))
    : []; // If no predictions, default to an empty array

  return (
    <main className="min-h-screen bg-white p-8 font-inter"> {/* Added font-inter for consistent font */}
      <div className="mx-auto max-w-[90%] bg-white p-8 rounded-xl shadow-lg"> {/* Added styling to main content div */}
        <div className="mb-12 text-center">
          <h1 className="mb-4 text-4xl font-bold tracking-tight text-gray-900">CNN Audio Visualizer</h1> {/* Changed text color */}
          <p className="mb-7 text-md font-light tracking-tight text-gray-700">Upload a WAV file to see model predictions and feature maps.</p> {/* Changed text color */}

          <div className="flex flex-col items-center gap-4"> {/* Space between elements in this column */}

            {/* Hidden file input element */}
            <input
              type="file"
              accept=".wav"
              id="file-upload"
              disabled={isLoading} // Disable input when loading
              className="hidden"
              onChange={handleFileChange} // Attach the file change handler
            />

            {/* Visually styled label that acts as the "Choose File" button */}
            <label
              htmlFor="file-upload"
              // Dynamically apply classes for visual feedback during loading
              className={`cursor-pointer bg-blue-600 hover:bg-blue-700 text-white font-bold py-2 px-6 rounded-lg shadow-md transition duration-300 ease-in-out
                ${isLoading
                  ? 'opacity-50 pointer-events-none' // Faded and unclickable when loading
                  : 'transform hover:scale-105'     // Scale effect on hover when not loading
                }`}
            >
              {/* Conditional text for the button */}
              {isLoading ? "Analysing...." : "Choose WAV File"}
            </label>

            {/* Display selected file name */}
            {fileName && !isLoading && ( // Show name only if not loading
              <p className="text-sm text-gray-600 mt-2">Selected: <span className="font-medium">{fileName}</span></p>
            )}

  
            {/* Display error message */}
            {error && (
              <p className="text-red-600 text-center font-medium mt-2">Error: {error}</p>
            )}

            {/* Section to display analysis results from the backend */}
            {vizData && (
              <div className="mt-8 p-6 bg-gray-50 rounded-lg shadow-inner w-full text-left">
                <h2 className="text-2xl font-semibold mb-6 text-gray-800 text-center">Analysis Results</h2>

                {/* Probability Chart for Top 3 Predictions */}
                {chartData.length > 0 && (
                  <div className="mb-6 bg-white p-4 rounded-lg shadow-md">
                    <h3 className="text-xl font-medium mb-4 text-gray-700 text-center">Top 3 Predictions</h3>
                    <ResponsiveContainer width="100%" height={300}>
                      <BarChart
                        data={chartData}
                        margin={{ top: 20, right: 30, left: 20, bottom: 5 }}
                      >
                        <CartesianGrid strokeDasharray="3 3" stroke="#e0e0e0" /> {/* Light gray grid lines */}
                        <XAxis dataKey="name" tick={{ fill: '#4a5568', fontSize: 12 }} /> {/* Darker tick labels */}
                        <YAxis tickFormatter={(value) => `${value}%`} domain={[0, 100]} tick={{ fill: '#4a5568', fontSize: 12 }} />
                        <Tooltip
                          // Custom formatter for tooltip content
                          formatter={(value) => [`${value}%`, 'Probability']}
                          labelFormatter={(label) => `Class: ${label}`}
                          // Custom styling for the tooltip box
                          contentStyle={{
                            backgroundColor: '#fff',
                            border: '1px solid #ccc',
                            borderRadius: '8px',
                            padding: '10px',
                            boxShadow: '0px 2px 10px rgba(0,0,0,0.1)'
                          }}
                          labelStyle={{ color: '#333', fontWeight: 'bold' }} // Styling for the label in tooltip
                          itemStyle={{ color: '#555' }} // Styling for the item value in tooltip
                        />
                        <Legend wrapperStyle={{ paddingTop: '10px' }} /> {/* Space above the legend */}
                        <Bar dataKey="Probability" fill="#3b82f6" radius={[10, 10, 0, 0]} /> {/* Blue bars with rounded top corners */}
                      </BarChart>
                    </ResponsiveContainer>
                  </div>
                )}

                {/* Display All Predictions (as before, but formatted) */}
                {vizData.predictions && vizData.predictions.length > 0 && (
                  <div className="mb-4">
                    <h3 className="text-xl font-medium mb-2 text-gray-700">All Predictions:</h3>
                    <ul className="list-disc list-inside text-gray-600">
                      {vizData.predictions.map((pred, index) => (
                        <li key={index}>
                          {pred.class.replace(/_/g, ' ')}: {(pred.confidence * 100).toFixed(2)}%
                        </li>
                      ))}
                    </ul>
                  </div>
                )}

                {/* Display Visualization Data (e.g., base64 image or plot data) */}
                {vizData.visualization && (
                  <div className="mb-4">
                    <h3 className="text-xl font-medium mb-2 text-gray-700">Visualization:</h3>
                    {/* Assuming visualization might be a base64 image data URI */}
                    {typeof vizData.visualization === 'string' && vizData.visualization.startsWith('data:image/') ? (
                      <img src={vizData.visualization} alt="Visualization Plot" className="max-w-full h-auto rounded-md shadow-sm" />
                    ) : (
                      // Fallback for other types of visualization data (e.g., JSON)
                      <pre className="bg-white p-3 rounded text-sm overflow-auto max-h-60 border border-gray-200">{JSON.stringify(vizData.visualization, null, 2)}</pre>
                    )}
                  </div>
                )}

                {/* Display Input Spectrogram Metadata (actual rendering would be complex) */}
                {vizData.input_spectrogram && (
                  <div className="mb-4">
                    <h3 className="text-xl font-medium mb-2 text-gray-700">Input Spectrogram Data:</h3>
                    <p className="text-gray-600">Shape: {vizData.input_spectrogram.shape.join('x')}</p>
                    {/* Placeholder for actual spectrogram visualization (e.g., using Canvas) */}
                  </div>
                )}

                {/* Display Waveform Metadata (actual rendering/playback would be complex) */}
                {vizData.waveform && (
                  <div>
                    <h3 className="text-xl font-medium mb-2 text-gray-700">Waveform Data:</h3>
                    <p className="text-gray-600">Sample Rate: {vizData.waveform.sample_rate} Hz</p>
                    <p className="text-gray-600">Duration: {vizData.waveform.duration.toFixed(2)} seconds</p>
                    {/* Placeholder for actual waveform visualization or audio playback */}
                  </div>
                )}
              </div>
            )}
          </div>
        </div>
      </div>
    </main>
  );
}
