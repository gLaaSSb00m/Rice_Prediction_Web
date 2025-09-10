import 'package:flutter/material.dart';
import 'package:http/http.dart' as http;
import 'package:image_picker/image_picker.dart';
import 'package:permission_handler/permission_handler.dart';
import 'package:flutter/foundation.dart' show kIsWeb, kDebugMode;
import 'package:http_parser/http_parser.dart';
import 'dart:io';
import 'dart:convert';

void main() {
  runApp(const RicePredictionApp());
}

class RicePredictionApp extends StatelessWidget {
  const RicePredictionApp({super.key});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'Rice Prediction',
      theme: ThemeData(
        colorScheme: ColorScheme.fromSeed(seedColor: Colors.green),
        useMaterial3: true,
      ),
      home: const HomePage(),
      debugShowCheckedModeBanner: false, // Disable debug banner here
    );
  }
}

class HomePage extends StatefulWidget {
  const HomePage({super.key});

  @override
  State<HomePage> createState() => _HomePageState();
}

class _HomePageState extends State<HomePage> {
  File? _image;
  String _predictedVariety = '';
  String _riceInfo = '';
  bool _isLoading = false;
  String _errorMessage = '';

  final ImagePicker _picker = ImagePicker();

  // Replace with your Django backend URL
  final String apiUrl = const String.fromEnvironment(
    'API_URL',
    defaultValue: 'http://127.0.0.1:8000/api/predict/',
  );
  // For Android emulator: 'http://10.0.2.2:8000/api/predict/'
  // For iOS simulator: 'http://localhost:8000/api/predict/'
  // For physical device: replace with your computer's IP address
  // Or set environment variable: flutter run --dart-define=API_URL=http://your-ip:8000/api/predict/

  Future<void> _requestPermissions() async {
    var cameraStatus = await Permission.camera.status;
    if (!cameraStatus.isGranted) {
      cameraStatus = await Permission.camera.request();
    }

    var photosStatus = await Permission.photos.status;
    if (!photosStatus.isGranted) {
      photosStatus = await Permission.photos.request();
    }

    // Check if permissions were granted
    if (!cameraStatus.isGranted && !photosStatus.isGranted) {
      setState(() {
        _errorMessage = 'Camera and gallery permissions are required to use this app';
      });
    }
  }

  Future<void> _pickImage(ImageSource source) async {
    try {
      final XFile? pickedFile = await _picker.pickImage(
        source: source,
        maxWidth: 1024,
        maxHeight: 1024,
        imageQuality: 85,
      );
      if (pickedFile != null) {
        // Validate file size (max 10MB)
        final fileSize = await pickedFile.length();
        if (fileSize > 10 * 1024 * 1024) {
          setState(() {
            _errorMessage = 'Image file is too large. Please select an image smaller than 10MB.';
          });
          return;
        }

        // Validate file type
        final fileName = pickedFile.name.toLowerCase();
        if (!fileName.endsWith('.jpg') && !fileName.endsWith('.jpeg') &&
            !fileName.endsWith('.png') && !fileName.endsWith('.bmp')) {
          setState(() {
            _errorMessage = 'Please select a valid image file (JPG, PNG, BMP).';
          });
          return;
        }

        setState(() {
          _image = File(pickedFile.path);
          _predictedVariety = '';
          _riceInfo = '';
          _errorMessage = '';
        });
      }
    } catch (e) {
      setState(() {
        _errorMessage = 'Error picking image: $e';
      });
    }
  }

  Future<void> _predictRice() async {
    if (_image == null) {
      setState(() {
        _errorMessage = 'Please select an image first';
      });
      return;
    }

    setState(() {
      _isLoading = true;
      _errorMessage = '';
    });

    try {
      if (kIsWeb) {
        // For web, use MultipartRequest with bytes and set content-type explicitly
        var request = http.MultipartRequest('POST', Uri.parse(apiUrl));
        var bytes = await _image!.readAsBytes();
        var multipartFile = http.MultipartFile.fromBytes(
          'rice_image',
          bytes,
          filename: 'upload.jpg',
          contentType: MediaType('image', 'jpg'),
        );
        request.files.add(multipartFile);

        var response = await request.send().timeout(const Duration(seconds: 30));
        var responseData = await response.stream.bytesToString();

        if (kDebugMode) {
          print('Response: $responseData');
        }

        if (response.statusCode == 200) {
          try {
            var jsonResponse = json.decode(responseData);

            // Validate response structure
            if (jsonResponse.containsKey('predicted_variety') && jsonResponse.containsKey('rice_info')) {
              setState(() {
                _predictedVariety = jsonResponse['predicted_variety']?.toString() ?? 'Unknown';
                _riceInfo = jsonResponse['rice_info']?.toString() ?? 'No information available';
              });
            } else {
              setState(() {
                _errorMessage = 'Invalid response format from server';
              });
            }
          } catch (e) {
            setState(() {
              _errorMessage = 'Failed to parse server response: $e';
            });
          }
        } else if (response.statusCode == 400) {
          setState(() {
            _errorMessage = 'Invalid image format. Please try a different image.';
          });
        } else if (response.statusCode == 413) {
          setState(() {
            _errorMessage = 'Image file is too large for processing.';
          });
        } else if (response.statusCode >= 500) {
          setState(() {
            _errorMessage = 'Server error. Please try again later.';
          });
        } else {
          try {
            var jsonResponse = json.decode(responseData);
            setState(() {
              _errorMessage = jsonResponse['error'] ?? 'Prediction failed (${response.statusCode})';
            });
          } catch (e) {
            setState(() {
              _errorMessage = 'Server error (${response.statusCode}). Please try again.';
            });
          }
        }
      } else {
        // For mobile platforms
        var request = http.MultipartRequest('POST', Uri.parse(apiUrl));
        request.files.add(await http.MultipartFile.fromPath('rice_image', _image!.path));

        var response = await request.send().timeout(const Duration(seconds: 30));
        var responseData = await response.stream.bytesToString();

        if (response.statusCode == 200) {
          try {
            var jsonResponse = json.decode(responseData);

            // Validate response structure
            if (jsonResponse.containsKey('predicted_variety') && jsonResponse.containsKey('rice_info')) {
              setState(() {
                _predictedVariety = jsonResponse['predicted_variety']?.toString() ?? 'Unknown';
                _riceInfo = jsonResponse['rice_info']?.toString() ?? 'No information available';
              });
            } else {
              setState(() {
                _errorMessage = 'Invalid response format from server';
              });
            }
          } catch (e) {
            setState(() {
              _errorMessage = 'Failed to parse server response: $e';
            });
          }
        } else if (response.statusCode == 400) {
          setState(() {
            _errorMessage = 'Invalid image format. Please try a different image.';
          });
        } else if (response.statusCode == 413) {
          setState(() {
            _errorMessage = 'Image file is too large for processing.';
          });
        } else if (response.statusCode >= 500) {
          setState(() {
            _errorMessage = 'Server error. Please try again later.';
          });
        } else {
          try {
            var jsonResponse = json.decode(responseData);
            setState(() {
              _errorMessage = jsonResponse['error'] ?? 'Prediction failed (${response.statusCode})';
            });
          } catch (e) {
            setState(() {
              _errorMessage = 'Server error (${response.statusCode}). Please try again.';
            });
          }
        }
      }
    } catch (e) {
      setState(() {
        // Show the error message with the exception details
        _errorMessage = 'Network error: $e';
      });
    } finally {
      setState(() {
        _isLoading = false;
      });
    }
  }

  @override
  void initState() {
    super.initState();
    _requestPermissions();
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: const Text('Rice Variety Prediction'),
        backgroundColor: Theme.of(context).colorScheme.inversePrimary,
      ),
      body: SingleChildScrollView(
        padding: const EdgeInsets.all(16.0),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.stretch,
          children: [
            const Text(
              'Upload Rice Image',
              style: TextStyle(fontSize: 24, fontWeight: FontWeight.bold),
              textAlign: TextAlign.center,
            ),
            const SizedBox(height: 20),

            // Image display
            Container(
              height: 200,
              decoration: BoxDecoration(
                border: Border.all(color: Colors.grey),
                borderRadius: BorderRadius.circular(10),
              ),
              child: _image != null
                  ? ClipRRect(
                      borderRadius: BorderRadius.circular(10),
                      child: 
                      // Use Image.network for web, Image.file for mobile
                      kIsWeb
                        ? Image.network(_image!.path, fit: BoxFit.cover)
                        : Image.file(_image!, fit: BoxFit.cover),
                    )
                  : const Center(
                      child: Text('No image selected'),
                    ),
            ),

            const SizedBox(height: 20),

            // Image selection buttons
            Row(
              mainAxisAlignment: MainAxisAlignment.spaceEvenly,
              children: [
                ElevatedButton.icon(
                  onPressed: () => _pickImage(ImageSource.camera),
                  icon: const Icon(Icons.camera),
                  label: const Text('Camera'),
                ),
                ElevatedButton.icon(
                  onPressed: () => _pickImage(ImageSource.gallery),
                  icon: const Icon(Icons.photo_library),
                  label: const Text('Gallery'),
                ),
              ],
            ),

            const SizedBox(height: 20),

            // Predict button
            ElevatedButton(
              onPressed: _isLoading ? null : _predictRice,
              style: ElevatedButton.styleFrom(
                padding: const EdgeInsets.symmetric(vertical: 16),
                backgroundColor: Colors.green,
              ),
              child: _isLoading
                  ? const CircularProgressIndicator()
                  : const Text('Predict Rice Variety'),
            ),

            const SizedBox(height: 20),

            // Error message
            if (_errorMessage.isNotEmpty)
              Container(
                padding: const EdgeInsets.all(12),
                decoration: BoxDecoration(
                  color: Colors.red.shade100,
                  borderRadius: BorderRadius.circular(8),
                ),
                child: Text(
                  _errorMessage,
                  style: const TextStyle(color: Colors.red),
                  textAlign: TextAlign.center,
                ),
              ),

            // Prediction results
            if (_predictedVariety.isNotEmpty) ...[
              const SizedBox(height: 20),
              const Text(
                'Prediction Result',
                style: TextStyle(fontSize: 20, fontWeight: FontWeight.bold),
                textAlign: TextAlign.center,
              ),
              const SizedBox(height: 10),
              Card(
                elevation: 4,
                child: Padding(
                  padding: const EdgeInsets.all(16),
                  child: Column(
                    crossAxisAlignment: CrossAxisAlignment.start,
                    children: [
                      Text(
                        'Rice Variety: $_predictedVariety',
                        style: const TextStyle(
                          fontSize: 18,
                          fontWeight: FontWeight.bold,
                          color: Colors.green,
                        ),
                      ),
                      const SizedBox(height: 10),
                      const Text(
                        'Information:',
                        style: TextStyle(fontWeight: FontWeight.bold),
                      ),
                      const SizedBox(height: 5),
                      Text(_riceInfo),
                    ],
                  ),
                ),
              ),
            ],
          ],
        ),
      ),
    );
  }
}
