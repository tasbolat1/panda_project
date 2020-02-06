cmake_minimum_required(VERSION 2.8.3)
project(panda_project_robot_pc)

add_compile_options(-std=c++11)

find_package(Eigen3 REQUIRED)

# Eigen 3.2 (Wily) only provides EIGEN3_INCLUDE_DIR, not EIGEN3_INCLUDE_DIRS
if(NOT EIGEN3_INCLUDE_DIRS)
  set(EIGEN3_INCLUDE_DIRS ${EIGEN3_INCLUDE_DIR})
endif()

#find_package(yaml-cpp REQUIRED)
find_package(catkin REQUIRED
  COMPONENTS
    controller_interface
    franka_gripper
    franka_msgs
    hardware_interface
    moveit_ros_planning
    moveit_ros_planning_interface
    moveit_visual_tools	
    pluginlib
    geometric_shapes
    rosbag
    tf2_ros
    tf2_eigen
    tf2_geometry_msgs
)

find_package(Boost REQUIRED system filesystem date_time thread)

catkin_package(
  LIBRARIES
    interactivity_utils
  INCLUDE_DIRS
    ${THIS_PACKAGE_INCLUDE_DIRS}
  CATKIN_DEPENDS
  DEPENDS
    EIGEN3
)

include_directories(SYSTEM ${THIS_PACKAGE_INCLUDE_DIRS} ${Boost_INCLUDE_DIR} ${EIGEN3_INCLUDE_DIRS})
include_directories(${catkin_INCLUDE_DIRS})
link_directories(${catkin_LIBRARY_DIRS})

#add_executable(simple_demo src/test1.cpp)
#add_executable(simple_demo_old src/test1_old.cpp)
#add_executable(move_grasp src/move_grasp.cpp)
#add_executable(grasp src/grasp.cpp)
#add_executable(get_pose src/get_pose.cpp)

#target_link_libraries(simple_demo ${catkin_LIBRARIES} ${Boost_LIBRARIES})
#target_link_libraries(simple_demo_old ${catkin_LIBRARIES} ${Boost_LIBRARIES})
#target_link_libraries(grasp ${catkin_LIBRARIES} ${Boost_LIBRARIES})
#target_link_libraries(move_grasp ${catkin_LIBRARIES} ${Boost_LIBRARIES} yaml-cpp)
#target_link_libraries(get_pose ${catkin_LIBRARIES} ${Boost_LIBRARIES})

#add_executable(gripper_client src/gripper_test1.cpp)
#add_executable(gripper_client_zero src/gripper_test2.cpp)
#target_link_libraries(gripper_client ${catkin_LIBRARIES})
#target_link_libraries(gripper_client_zero ${catkin_LIBRARIES})


#add_executable(homing src/homing.cpp)
#target_link_libraries(homing ${catkin_LIBRARIES})

#add_dependencies(
#  gripper_client
#  ${panda_examples_EXPORTED_TARGETS}
#)


