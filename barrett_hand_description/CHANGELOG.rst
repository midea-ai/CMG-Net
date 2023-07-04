^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Changelog for package barrett_hand_description
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

0.1.2 (2016-08-10)
------------------

0.1.1 (2016-08-09)
------------------
* cleaned up bh280 mimic joints
* change bh280 to mimic joints
* updated gazebo elements to support multiples/renaming
  -added ${name} param to mimic joint names to match corresponding urdf
  -moved control plugin to bh_alone.urdf.xacro so multiple hands can be included into other robots without a name collision in gazebo
* Adding Changelog files
* Url fix
* Fixing catkin error
* Adding metapackage and setting CMakeLists and package.xml for release
* Add Barrett Hand description package
* Contributors: Allison Thackston, Elena Gambaro, RomanRobotnik
