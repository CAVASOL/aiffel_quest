// import 'package:flutter/material.dart';

// void main() {
//   runApp(const MyApp());
// }

// class MyApp extends StatelessWidget {
//   const MyApp({Key? key}) : super(key: key);

//   onChanged() {
//     print('HAHAHAHAHAHAHAHAH');
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
//                   onPressed: onChanged,
//                   child: const Text(
//                     'Text',
//                     style: TextStyle(
//                       fontSize: 24,
//                     ),
//                   ),
//                 ),
//                 const SizedBox(
//                   height: 20,
//                 ),
//                 Stack(
//                   children: [
//                     Container(
//                       width: 350,
//                       height: 350,
//                       color: Colors.deepPurple,
//                     ),
//                     Container(
//                       width: 300,
//                       height: 300,
//                       color: Colors.blueGrey,
//                     ),
//                     Container(
//                       width: 250,
//                       height: 250,
//                       color: Colors.lightGreen,
//                     ),
//                     Container(
//                       width: 200,
//                       height: 200,
//                       color: Colors.amber,
//                     ),
//                     Container(
//                       width: 150,
//                       height: 150,
//                       color: Colors.orange,
//                     ),
//                     Container(
//                       width: 100,
//                       height: 100,
//                       color: Colors.deepOrange,
//                     ),
//                     Container(
//                       width: 50,
//                       height: 50,
//                       color: Colors.pink,
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
