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
//       debugShowCheckedModeBanner: false,
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
//   String result = "";
//   late TextEditingController urlController;

//   @override
//   void initState() {
//     super.initState();
//     urlController = TextEditingController();
//   }

//   Future<void> fetchData(String apiUrl) async {
//     try {
//       final response = await http.get(
//         Uri.parse(apiUrl),
//         headers: {
//           'Content-Type': 'application/json',
//           'ngrok-skip-browser-warning': '69420',
//         },
//       );
//       if (response.statusCode == 200) {
//         final data = jsonDecode(response.body);
//         setState(() {
//           result = "Result: $data";
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
//   void dispose() {
//     urlController.dispose();
//     super.dispose();
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
//                     onPressed: () => fetchData(
//                         'https://3bd1-34-83-37-72.ngrok-free.app/sample1'),
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
//                     onPressed: () => fetchData(
//                         'https://3bd1-34-83-37-72.ngrok-free.app/sample2'),
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
