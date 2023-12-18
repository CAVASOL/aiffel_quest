// // ignore_for_file: must_be_immutable, non_constant_identifier_names

// import 'package:flutter/material.dart';

// void main() {
//   runApp(const MyApp());
// }

// class MyApp extends StatelessWidget {
//   const MyApp({Key? key}) : super(key: key);

//   @override
//   Widget build(BuildContext context) {
//     return MaterialApp(
//       debugShowCheckedModeBanner: false,
//       initialRoute: '/first',
//       routes: {
//         '/first': (context) => FirstScreen(),
//         '/second': (context) => SecondScreen(),
//       },
//     );
//   }
// }

// class FirstScreen extends StatelessWidget {
//   FirstScreen({Key? key}) : super(key: key);

//   bool is_cat = true;

//   void onChanged() {
//     if (is_cat) {
//       print("Cat's state: $is_cat");
//     } else {
//       return;
//     }
//   }

//   @override
//   Widget build(BuildContext context) {
//     return MaterialApp(
//       debugShowCheckedModeBanner: false,
//       home: Scaffold(
//         appBar: AppBar(
//           leading: Image.asset(
//             'images/caticon.png',
//           ),
//           title: const Text("First Page"),
//         ),
//         body: Container(
//           color: Colors.white,
//           child: Center(
//             child: Column(
//               mainAxisAlignment: MainAxisAlignment.center,
//               children: [
//                 ElevatedButton(
//                   onPressed: () {
//                     onChanged();
//                     Navigator.pushNamed(context, '/second');
//                   },
//                   child: const Text(
//                     "Next",
//                   ),
//                 ),
//                 Image.asset(
//                   'images/cat.png',
//                   width: 300,
//                   height: 300,
//                 )
//               ],
//             ),
//           ),
//         ),
//       ),
//     );
//   }
// }

// class SecondScreen extends StatelessWidget {
//   SecondScreen({Key? key}) : super(key: key);

//   bool is_cat = false; // Changed the state to false for demonstration

//   void onChanged() {
//     if (!is_cat) {
//       print("Cat's state: $is_cat");
//     } else {
//       return;
//     }
//   }

//   @override
//   Widget build(BuildContext context) {
//     return MaterialApp(
//       debugShowCheckedModeBanner: false,
//       home: Scaffold(
//         appBar: AppBar(
//           leading: Image.asset(
//             'images/dogicon.png',
//           ),
//           title: const Text("Second Page"),
//         ),
//         body: Container(
//           color: Colors.white,
//           child: Center(
//             child: Column(
//               mainAxisAlignment: MainAxisAlignment.center,
//               children: [
//                 ElevatedButton(
//                   onPressed: () {
//                     onChanged();
//                     Navigator.pop(context);
//                   },
//                   child: const Text('Back'),
//                 ),
//                 Image.asset(
//                   'images/dog.png',
//                   width: 300,
//                   height: 300,
//                 )
//               ],
//             ),
//           ),
//         ),
//       ),
//     );
//   }
// }
