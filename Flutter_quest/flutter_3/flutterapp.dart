// // ignore_for_file: avoid_print

// import 'package:flutter/material.dart';

// void main() {
//   runApp(const MyApp());
// }

// class MyApp extends StatelessWidget {
//   const MyApp({Key? key}) : super(key: key);

//   onChanged() {
//     print('HAHAHAHAHAHAHAHAHAHAHAHAHAHAHAH');
//   }

//   @override
//   Widget build(BuildContext context) {
//     return MaterialApp(
//       debugShowCheckedModeBanner: false,
//       home: Scaffold(
//         appBar: AppBar(
//           leading: const Icon(
//             Icons.favorite,
//           ),
//           title: const Text(
//             'Flutter App',
//             style: TextStyle(
//               color: Colors.white,
//             ),
//           ),
//           backgroundColor: Colors.black,
//         ),
//         body: Container(
//           margin: const EdgeInsets.all(
//             20,
//           ),
//           child: Center(
//             child: Column(
//               children: [
//                 ElevatedButton(
//                   style: ElevatedButton.styleFrom(
//                     backgroundColor: Colors.black,
//                   ),
//                   onPressed: onChanged,
//                   child: const Text(
//                     'Text',
//                     style: TextStyle(
//                       fontSize: 20,
//                     ),
//                   ),
//                 ),
//                 const SizedBox(
//                   height: 20,
//                 ),
//                 Stack(
//                   children: [
//                     Container(
//                       width: 300,
//                       height: 300,
//                       color: Colors.deepPurple,
//                     ),
//                     Container(
//                       width: 240,
//                       height: 240,
//                       color: Colors.blueGrey,
//                     ),
//                     Container(
//                       width: 180,
//                       height: 180,
//                       color: Colors.lightGreen,
//                     ),
//                     Container(
//                       width: 120,
//                       height: 120,
//                       color: Colors.amber,
//                     ),
//                     Container(
//                       width: 60,
//                       height: 60,
//                       color: Colors.pinkAccent,
//                     ),
//                   ],
//                 )
//               ],
//             ),
//           ),
//         ),
//       ),
//     );
//   }
// }
