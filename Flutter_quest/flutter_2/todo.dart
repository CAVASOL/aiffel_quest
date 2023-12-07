import 'dart:io';

class ToDoList {
  List<Map<String, dynamic>> tasks = [];

  void addTask(String task) {
    DateTime formattedDate = DateTime.now();

    Map<String, dynamic> newTask = {
      'task': task,
      'date': formattedDate.toIso8601String(),
      'time': formattedDate.toLocal().toString(),
    };

    tasks.add(newTask);
    print(
        'Task "$task" has been added to the to-do list at ${formattedDate.toLocal()} on ${formattedDate.toIso8601String()}');
  }

  void removeTask(int index) {
    if (index >= 0 && index < tasks.length) {
      String removedTask = tasks[index]['task'];
      tasks.removeAt(index);
      print('Task "$removedTask" has been removed from the to-do list.');
    } else {
      print('Invalid index. Please enter a valid task index to remove.');
    }
  }

  void displayTasks() {
    if (tasks.isEmpty) {
      print('The to-do list is empty.');
    } else {
      print('Tasks in the to-do list:');
      for (var i = 0; i < tasks.length; i++) {
        print(
            '${i + 1}. ${tasks[i]['task']} - Added on ${tasks[i]['date']} at ${tasks[i]['time']}');
      }
    }
  }
}

void main() {
  var toDoList = ToDoList();

  while (true) {
    print('\nChoose an option:');
    print('1. Add a task');
    print('2. Remove a task');
    print('3. Display tasks');
    print('4. Exit');

    var choice = int.tryParse(stdin.readLineSync() ?? '');

    switch (choice) {
      case 1:
        print('Enter the task to add:');
        var task = stdin.readLineSync();
        if (task != null && task.isNotEmpty) {
          toDoList.addTask(task);
        } else {
          print('Please enter a valid task.');
        }
        break;
      case 2:
        print('Enter the index of the task to remove:');
        var index = int.tryParse(stdin.readLineSync() ?? '');
        if (index != null) {
          toDoList.removeTask(index - 1);
        } else {
          print('Please enter a valid index.');
        }
        break;
      case 3:
        toDoList.displayTasks();
        break;
      case 4:
        print('See you next time!');
        return;
      default:
        print('Please enter a valid option.');
    }
  }
}
