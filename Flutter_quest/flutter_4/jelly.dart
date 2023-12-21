// // ignore_for_file: use_key_in_widget_constructors, library_private_types_in_public_api, avoid_print

// import 'dart:convert';
// import 'package:flutter/material.dart';
// import 'package:http/http.dart' as http;

// void main() {
//   runApp(const MyApp());
// }

// class MyApp extends StatelessWidget {
//   const MyApp({Key? key});

//   @override
//   Widget build(BuildContext context) {
//     return const MaterialApp(
//       home: MyHomePage(),
//     );
//   }
// }

// class MyHomePage extends StatefulWidget {
//   const MyHomePage({Key? key});

//   @override
//   _MyHomePageState createState() => _MyHomePageState();
// }

// class _MyHomePageState extends State<MyHomePage> {
//   TextEditingController urlController = TextEditingController();

//   Future<void> fetchUrlData(String url) async {
//     try {
//       final response = await http.get(
//         Uri.parse(url),
//         headers: {
//           'Content-Type': 'application/json',
//           'ngrok-skip-browser-warning': '69420',
//         },
//       );
//       if (response.statusCode == 200) {
//         final data = jsonDecode(response.body);
//         print("URL: $url");
//         print("predicted_label: ${data['predicted_label']}");
//         print("prediction_score: ${data['prediction_score']}");
//       } else {
//         print(
//             "Failed to fetch data from $url. Status Code: ${response.statusCode}");
//       }
//     } catch (e) {
//       print("Error fetching data from $url: $e");
//     }
//   }

//   @override
//   Widget build(BuildContext context) {
//     return Scaffold(
//       appBar: AppBar(
//         leading: Image.asset(
//           'images/jelly.png',
//         ),
//         title: const Text(
//           'Jellyfish Classifier',
//           style: TextStyle(
//             color: Colors.black,
//             fontWeight: FontWeight.w600,
//             fontSize: 20,
//           ),
//         ),
//         backgroundColor: Colors.white,
//       ),
//       resizeToAvoidBottomInset: false,
//       body: Center(
//         child: Column(
//           mainAxisAlignment: MainAxisAlignment.center,
//           children: [
//             Image.asset(
//               'images/jelly.jpg',
//               width: 300,
//               height: 300,
//             ),
//             const SizedBox(
//               height: 10,
//             ),
//             Row(
//               mainAxisAlignment: MainAxisAlignment.spaceAround,
//               children: [
//                 ElevatedButton(
//                   style: ElevatedButton.styleFrom(
//                     backgroundColor: Colors.cyan,
//                   ),
//                   onPressed: () => fetchUrlData(
//                       "https://9e18-34-83-37-72.ngrok-free.app/sample1"),
//                   child: const Text(
//                     'Label',
//                     style: TextStyle(
//                       fontSize: 16,
//                       color: Colors.white,
//                     ),
//                   ),
//                 ),
//                 ElevatedButton(
//                   style: ElevatedButton.styleFrom(
//                     backgroundColor: Colors.cyan,
//                   ),
//                   onPressed: () => fetchUrlData(
//                       "https://9e18-34-83-37-72.ngrok-free.app/sample2"),
//                   child: const Text(
//                     'Score',
//                     style: TextStyle(
//                       fontSize: 16,
//                       color: Colors.white,
//                     ),
//                   ),
//                 ),
//               ],
//             ),
//           ],
//         ),
//       ),
//     );
//   }
// }
