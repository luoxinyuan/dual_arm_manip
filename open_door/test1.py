from arm import Arm

#from arm_package.robotic_arm import Arm as ArmBase

from base import Base


arm_left = Arm.init_from_yaml(cfg_path='/cfg/cfg_arm_left.yaml')

# print('==========\nArm Connecting...')
#         self.arm = ArmBase(self.host_ip,self.host_port)
#         self.change_tool_frame(self.tool_frame)
#         print('Arm Connected\n==========')


#print("arm_left: ", arm_left)

print("arm get pos", arm_left.get_p())

#print("arm ik",arm_left.ik([0.7207090258598328, -0.2044149935245514, 0.2483610063791275, -1.5809999704360962, 0.515999972820282, -2.0369999408721924]))


#arm_left.run()