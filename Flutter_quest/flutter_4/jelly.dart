// // ignore_for_file: library_private_types_in_public_api, avoid_print

// import 'dart:convert';
// import 'package:flutter/material.dart';
// import 'package:http/http.dart' as http;

// void main() {
//   runApp(const MyApp());
// }

// class MyApp extends StatelessWidget {
//   const MyApp({super.key});

//   @override
//   Widget build(BuildContext context) {
//     return const MaterialApp(
//       debugShowCheckedModeBanner: false,
//       home: MyHomePage(),
//     );
//   }
// }

// class MyHomePage extends StatefulWidget {
//   const MyHomePage({super.key});

//   @override
//   _MyHomePageState createState() => _MyHomePageState();
// }


// class _MyHomePageState extends State<MyHomePage> {
//   String result = "";
//   TextEditingController urlController = TextEditingController();

//   Future<void> fetchData(int queryType) async {
//     try {
//       const enteredUrl = 'https://5647-34-83-37-72.ngrok-free.app/';
//       final response = await http.get(
//         Uri.parse("${enteredUrl}sample1"),
//         headers: {
//           'Content-Type': 'application/json',
//           'ngrok-skip-browser-warning': '69420',
//         },
//       );
//       if (response.statusCode == 200) {
//         final data = jsonDecode(response.body);
//         setState(() {
//           if (queryType == 1) {
//             result = "Predicted label: ${data['predicted_label']}";
//           } else if (queryType == 2) {
//             result = "Predicted score: ${data['prediction_score']}";
//           }
//         });
//       } else {
//         setState(() {
//           result = "Failed to fetch data. Status Code: ${response.statusCode}";
//         });
//       }
//       print(result);
//     } catch (e) {
//       setState(() {
//         result = "Error: $e";
//       });
//     }
//   }

//   @override
//   Widget build(BuildContext context) {
//     return MaterialApp(
//       debugShowCheckedModeBanner: false,
//       home: Scaffold(
//         appBar: AppBar(
//           leading: Image.asset(
//             'images/jelly.png',
//           ),
//           title: const Text(
//             'Jellyfish Classifier',
//             style: TextStyle(
//               color: Colors.black,
//               fontWeight: FontWeight.w600,
//               fontSize: 20,
//             ),
//           ),
//           backgroundColor: Colors.white,
//         ),
//         resizeToAvoidBottomInset: false,
//         body: Center(
//           child: Column(
//             mainAxisAlignment: MainAxisAlignment.center,
//             children: [
//               Image.asset(
//                 'images/jelly.jpg',
//                 width: 300,
//                 height: 300,
//               ),
//               const SizedBox(
//                 height: 10,
//               ),
//               Row(
//                 mainAxisAlignment: MainAxisAlignment.spaceAround,
//                 children: [
//                   ElevatedButton(
//                     style: ElevatedButton.styleFrom(
//                       backgroundColor: Colors.cyan,
//                     ),
//                     onPressed: () => fetchData(1),
//                     child: const Text(
//                       'Label',
//                       style: TextStyle(
//                         fontSize: 16,
//                         color: Colors.white,
//                       ),
//                     ),
//                   ),
//                   ElevatedButton(
//                     style: ElevatedButton.styleFrom(
//                       backgroundColor: Colors.cyan,
//                     ),
//                     onPressed: () => fetchData(2),
//                     child: const Text(
//                       'Score',
//                       style: TextStyle(
//                         fontSize: 16,
//                         color: Colors.white,
//                       ),
//                     ),
//                   ),
//                 ],
//               ),
//               const SizedBox(
//                 height: 10,
//               ),
//               Text(
//                 result,
//                 style: const TextStyle(
//                   fontSize: 16,
//                 ),
//               ),
//             ],
//           ),
//         ),
//       ),
//     );
//   }
// }
