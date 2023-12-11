import 'package:flutter/material.dart';
import 'dart:async';

void pomodoro() {
  runApp(const PomodoroApp());
}

class PomodoroApp extends StatelessWidget {
  const PomodoroApp({super.key});

  @override
  Widget build(BuildContext context) {
    return const MaterialApp(
      home: PomodoroTimer(),
    );
  }
}

class PomodoroTimer extends StatefulWidget {
  const PomodoroTimer({super.key});

  @override
  _PomodoroTimerState createState() => _PomodoroTimerState();
}

class _PomodoroTimerState extends State<PomodoroTimer> {
  int workDuration = 25 * 60;
  int breakDuration = 5 * 60;
  int cycles = 0;
  int currentDuration = 25 * 60;
  late Timer timer;
  bool isWorking = true;
  bool isPaused = true;

  @override
  void initState() {
    super.initState();
    startTimer();
  }

  void startTimer() {
    timer = Timer.periodic(const Duration(seconds: 1), (timer) {
      setState(() {});
      if (!isPaused) {
        if (currentDuration > 0) {
          currentDuration--;
        } else {
          if (isWorking) {
            cycles++;
            if (cycles >= 4) {
              workDuration = 25 * 60;
              breakDuration = 15 * 60;
              cycles = 0;
            }
            isWorking = false;
            currentDuration = breakDuration;
          } else {
            isWorking = true;
            currentDuration = workDuration;
          }
        }
      }
    });
  }

  void toggleTimer() {
    setState(() {});
    isPaused = !isPaused;
  }

  String getTimerText() {
    int minutes = (currentDuration ~/ 60);
    int seconds = (currentDuration % 60);
    String minutesStr = (minutes < 10) ? '0$minutes' : '$minutes';
    String secondsStr = (seconds < 10) ? '0$seconds' : '$seconds';
    return '$minutesStr:$secondsStr';
  }

  @override
  void dispose() {
    timer.cancel();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      home: Scaffold(
        backgroundColor: const Color.fromARGB(255, 230, 178, 22),
        body: Center(
          child: Column(
            mainAxisAlignment: MainAxisAlignment.center,
            children: [
              Text(
                isWorking ? 'Work Time' : 'Break Time',
                style: const TextStyle(
                  fontSize: 24,
                  fontWeight: FontWeight.w800,
                  color: Color(0xFF212847),
                ),
              ),
              const SizedBox(height: 20),
              Text(
                getTimerText(),
                style: const TextStyle(
                  fontSize: 48,
                  color: Color(0xFF212847),
                ),
              ),
              const SizedBox(height: 10),
              Center(
                child: Center(
                  child: IconButton(
                    iconSize: 80,
                    color: const Color(0xFFF92672),
                    onPressed: toggleTimer,
                    icon: Icon(isPaused
                        ? Icons.pause_circle_outline
                        : Icons.play_circle_outline),
                  ),
                ),
              ),
            ],
          ),
        ),
      ),
    );
  }
}
