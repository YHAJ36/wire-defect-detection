import sensor
import image
import lcd
import KPU as kpu
import time
from Maix import FPIOA, GPIO
import gc
from fpioa_manager import fm
from board import board_info
import utime
from machine import Timer,PWM
from machine import UART
from machine import WDT
import os
from machine import SPI

# 加载模型
task_fd = kpu.load(0x100000) # 从flash 0x100000 加载人脸检测模型
task_ld = kpu.load(0x200000) # 从flash 0x200000 加载人脸五点关键点检测模型
task_fe = kpu.load(0x300000) # 从flash 0x300000 加载人脸196维特征值模型

#####################配置蓝牙串口#####################
fm.register(30, fm.fpioa.UART1_TX, force=True)
fm.register(31, fm.fpioa.UART1_RX, force=True)

uart = UART(UART.UART1, 9600, 8, 1, 0, timeout=1000, read_buf_len=4096)

#####################配置蓝牙串口#####################
#fm.register(21, fm.fpioa.UART2_TX, force=True)
fm.register(21, fm.fpioa.UART2_RX, force=True)

uart_blue = UART(UART.UART2, 9600, 8, 1, 0, timeout=1000, read_buf_len=4096)

lcd.init()
sensor.reset()
sensor.set_pixformat(sensor.RGB565)
sensor.set_framesize(sensor.QVGA)
sensor.set_hmirror(1)
sensor.set_vflip(1)
sensor.run(1)
anchor = (1.889, 2.5245, 2.9465, 3.94056, 3.99987, 5.3658, 5.155437,
          6.92275, 6.718375, 9.01025)  # anchor for face detect
dst_point = [(44, 59), (84, 59), (64, 82), (47, 105),
             (81, 105)]  # standard face key point position
a = kpu.init_yolo2(task_fd, 0.5, 0.3, 5, anchor)
img_lcd = image.Image()
img_face = image.Image(size=(128, 128))
a = img_face.pix_to_ai()

image.font_load(image.UTF8, 16, 16, 0xA00000)

record_ftr = []
record_ftrs = []
#name = ['张小明', '王小红', '李小丽', '赵小聪', '黄小兰', '周小芳', '刘小贝']
#name = ['Mr.1', 'Mr.2', 'Mr.3', 'Mr.4', 'Mr.5']
names = []
record_ftrtemp=[]  #临时存储录入时的人脸特征

################## 开机时读取SD卡中的人脸信息 ##################
i = 0
temp_num = ''
with open("/sd/faceinfo.txt", "r") as f:
    while(1):
        lin = f.readline()
        if not lin:
            break
        stu_num = lin[0:lin.index('#')]    #获取学号
        names.append(stu_num)              #追加到学号列表
        faceftr = lin[lin.index('#')+1:]   #截取后半段的人脸特征
        record_ftrs.append(eval(faceftr))  #向人脸特征列表中添加SD卡中的已存特征
        if stu_num == temp_num:
            continue
        temp_num = stu_num
        i = i + 1
        print("%d:%s#%s" % (i,stu_num,faceftr))
    f.close()
print(names)


#####################配置按键####################
LuRu_mode = False   # 录入模式
Door_mode = True   # 门禁模式
del_all = False

#fpioa = FPIOA()
#fm.register(16, fm.fpioa.GPIOHS0)
#key_gpio = GPIO(GPIO.GPIOHS0, GPIO.IN)

#last_key_state=1
#key_pressed=0
#def check_key(): # 按键检测函数，用于在循环中检测按键是否按下，下降沿有效
    #global last_key_state
    #global key_pressed
    #global del_all

    #val=key_gpio.value()
    #if last_key_state == 1 and val == 0:
        #del_all = True
    #last_key_state = val


#####################配置看门狗#####################
#def on_wdt(self):
    #self.feed()
#wdt0 = WDT(id=1, timeout=5000, callback=on_wdt, context={})

###################################################################
##                            主循环                             ##
###################################################################

delline = []
check_num = 0
uart.read()

# 人脸对比置信度
ACCURACY = 80

luru_flag = 0
find_flag = 0

feature = ''
max_score = 0
index = 0

name_index = int(len(names)/3)
print('name_index: ', name_index)
shibie_num = 0

uart.write('{ok}')

uart.read()

while (1):
    gc.collect()
    #check_key()
    img = sensor.snapshot()

    code = kpu.run_yolo2(task_fd, img)
    max_score = 0
    if code:
        t = 0
        max_face = 0
        totalRes = len(code)
        area = []
        if totalRes > 1:  #多张人脸时，挑选面积最大的，一般即最前面的
            for i in code: # 迭代坐标框  多张人脸
                area.append(i.w()*i.h())
            for j in range(len(area)):
                if max_face < area[j]:
                    max_face = area[j]
                    t = j    #保存最大脸面积下标
            totalRes = 1
            del area
        if totalRes == 1:  #只有一张脸
            try:
                i = code[t]
                # Cut face and resize to 128x128
                a = img.draw_rectangle(i.rect())
                face_cut = img.cut(i.x(), i.y(), i.w(), i.h())
                face_cut_128 = face_cut.resize(128, 128)
                a = face_cut_128.pix_to_ai()
                # a = img.draw_image(face_cut_128, (0,0))
                # Landmark for face 5 points
                fmap = kpu.forward(task_ld, face_cut_128)
                plist = fmap[:]
                le = (i.x() + int(plist[0] * i.w() - 10), i.y() + int(plist[1] * i.h()))
                re = (i.x() + int(plist[2] * i.w()), i.y() + int(plist[3] * i.h()))
                nose = (i.x() + int(plist[4] * i.w()), i.y() + int(plist[5] * i.h()))
                lm = (i.x() + int(plist[6] * i.w()), i.y() + int(plist[7] * i.h()))
                rm = (i.x() + int(plist[8] * i.w()), i.y() + int(plist[9] * i.h()))
                a = img.draw_circle(le[0], le[1], 4)
                a = img.draw_circle(re[0], re[1], 4)
                a = img.draw_circle(nose[0], nose[1], 4)
                a = img.draw_circle(lm[0], lm[1], 4)
                a = img.draw_circle(rm[0], rm[1], 4)
                # align face to standard position
                src_point = [le, re, nose, lm, rm]
                T = image.get_affine_transform(src_point, dst_point)
                a = image.warp_affine_ai(img, img_face, T)
                a = img_face.ai_to_pix()
                # a = img.draw_image(img_face, (128,0))
                del (face_cut_128)
                # calculate face feature vector
                fmap = kpu.forward(task_fe, img_face)
                feature = kpu.face_encode(fmap[:])
                reg_flag = False
                scores = []
                for j in range(len(record_ftrs)):
                    score = kpu.face_compare(record_ftrs[j], feature)
                    scores.append(score)
                max_score = 0
                index = 0
                for k in range(len(scores)):
                    if max_score < scores[k]:
                        max_score = scores[k]
                        index = k
            except:
                pass

    # 接收 STM32 数据
    data = uart.read()
    if data != None and len(data) >= 2: #如果读取到了数据，且大于等于2
        # 切换
        print(data)
        if 'luru' in data:
            print(data)
            LuRu_mode = True
            Door_mode = False
            print('luru')
        if 'door' in data:
            Door_mode = True
            LuRu_mode = False
            print('door')
        if 'delall' in data:
            print('delall')
            del_all = True
    # 接收 蓝牙 数据
    data1 = uart_blue.read()
    if data1 != None and len(data1) >= 2: #如果读取到了数据，且大于等于2
        # luID
        if 'lu' in data1 and LuRu_mode == True:
            print(data1)
            people_num = data1[2:].decode('utf-8')   # 分割出ID
            print(people_num)
            luru_flag = 1   # 打卡录入任务
        if 'door' in data1:
            Door_mode = True
            LuRu_mode = False
            print('door')
        # 临时开门
        if 'open' in data1:
            print("--------蓝牙开门--------")
            uart.write('{ok}')
        if 'delall' in data1:
            print('delall')
            del_all = True
            # 删除用户命令
        if 'delete' in data1:
            lcd.clear()
            del_all = False
            print("--------删除用户--------")
            delnum = data1[6:].decode('utf-8')  #截取出删除的学号
            print('删除的学号: ', delnum)
            # 如果已录入此ID，那么就删除，
            if delnum in names:
                while delnum in names:
                    index = names.index(delnum)
                    names.pop(index)  # 删除ID
                    record_ftrs.pop(index)  # 删除特征值
                print('剩下: ',names)
                try:
                    with open("/sd/faceinfo.txt", "w") as f:
                        for i in range(len(record_ftrs)):  #循环遍历临时特征值列表，写入SD卡
                            f.write(names[i]+'#'+str(record_ftrs[i]))
                            f.write("\n")
                        f.close()
                except:
                    pass
                img.draw_rectangle((90, 85, 140, 70), fill=True, color=(0, 0, 255))
                img.draw_string(110, 112, "delete %s"%delnum, color=(255, 255, 255), scale=1.5, mono_space=0)
                lcd.display(img)
                time.sleep(2)
            else:
                a = img.draw_rectangle((90, 85, 140, 70), fill=True, color=(0, 0, 255))
                a = img.draw_string(118, 112, b"No people!", color=(255, 255, 255), scale=1.5, mono_space=0)
                a = lcd.display(img)
                time.sleep(2)
    # 执行录入任务
    if luru_flag == 1:
        a = img.draw_string(0,0, b'LuRu Mode', color=(0,255,0),scale=1,mono_space=1)
        if people_num in names:
            a = img.draw_string(160,0, b'ID Exist', color=(255,0,0),scale=1,mono_space=1)  #提示人脸已存在
        else:
            stu_num = people_num  # 取出人名
            a = img.draw_string(0,50, b'Start %s'%stu_num, color=(0,255,0),scale=1,mono_space=1) #显示开始录入
            if max_score > ACCURACY:
                check_num = 0
                record_ftrtemp.clear()
                a = img.draw_string(160,0, b'Face Exist', color=(255,0,0),scale=1,mono_space=1)  #提示人脸已存在
                a = lcd.display(img)
                continue
            if code:
                check_num = check_num + 1     #检测到一次人脸则加 1
                a = img.draw_string(5,25, b'%d'%check_num, color=(0,255,0),scale=1,mono_space=1) #显示采样进度
                #间隔录入
                if check_num % 6 == 0 and check_num != 0:
                    record_ftrtemp.append(feature)   #加入当前人脸特征
                #录入结束，总三个人脸特征值
                if check_num == 18:
                    people_num = ''
                    check_num = 0
                    save_success = 1  #录入成功标志位
                    try:
                        with open("/sd/faceinfo.txt", "a") as f:
                            for i in range(len(record_ftrtemp)):  #循环遍历临时特征值列表，写入SD卡
                                f.write(stu_num+'#'+str(record_ftrtemp[i]))
                                f.write("\n")
                            f.close()
                    except Exception:
                        save_success = 0  #表示保存到SD卡失败了
                        pass
                    #如果保存到SD卡成功，接着保存到全局数组
                    if save_success == 1:
                        #将当前特征添加到已知特征列表，学号添加到names
                        for i in record_ftrtemp:
                            record_ftrs.append(i)
                            names.append(stu_num)
                        name_index = name_index + 1
                        record_ftrtemp.clear()  #清空临时特征值列表
                        a = img.draw_rectangle((90, 85, 140, 70), fill=True, color=(0, 0, 255))
                        a = img.draw_string(100,110, b'Success!', color=(255,255,255),scale=1,mono_space=1)  #录入成功
                        a = lcd.display(img)
                        time.sleep(2)
                        luru_flag = 0  # 退出录入任务
                    #如果保存到SD卡失败，则按键次数清零，check_num清零，学号清空
                    else:
                        a = img.draw_rectangle((90, 85, 140, 70), fill=True, color=(0, 0, 255))
                        a = img.draw_string(110,110, b'Fail!', color=(255,255,255),scale=1,mono_space=1)  #录入失败
                        a = lcd.display(img)
                        time.sleep(2)

                        luru_flag = 0  # 退出录入任务
                    LuRu_mode = True

    # 检测按键
    # 录入模式按键
    if LuRu_mode:
        shibie_num = 0
        try:
            a = img.draw_string(0,0, b'LuRu Mode', color=(0,255,0),scale=1,mono_space=1)
        except:
            pass

    # 门禁模式按键
    if Door_mode:
        luru_flag = 0
        check_num = 0
        a = img.draw_string(0,0, b'Door Mode', color=(0,255,0),scale=1,mono_space=2)

        if code:
            if max_score > ACCURACY:
                a = img.draw_string(i.x()+4,i.y()-20, (b"%s: %2.1f" % (names[index], max_score)), color=(0,255,0),scale=1,mono_space=1) # 显示学号与分数
            else:
                a = img.draw_string(i.x()+4,i.y()-20, (b"%s: %2.1f" % ("No",max_score)), color=(255,0,0),scale=1,mono_space=1) # 显示分数

            shibie_num = shibie_num + 1
            if shibie_num == 5:
                shibie_num = 0
                try:
                    if max_score > ACCURACY:
                        a = img.draw_string(0,20, b'Success!', color=(0,255,0),scale=1,mono_space=1)  #录入成功
                        a = lcd.display(img)
                        print("--------人脸识别成功--------")
                        uart.write('{ok}')
                        time.sleep(2)
                    else:
                        uart.write('{fail}')
                        a = img.draw_string(0,20, b'Fail!', color=(255,0,0),scale=1,mono_space=1)  #录入失败
                        img.draw_string(i.x()+4,i.y()-20, (b"%s: %2.1f" % ("No",max_score)), color=(255,0,0),scale=1,mono_space=1) # 显示分数
                        a = lcd.display(img)
                        time.sleep(2)
                except:
                    pass
        else:
            shibie_num = 0

    # 删除全部特征值按键
    if del_all:
        del_all = False
        print("--------删除全部用户--------")
        name_index = 0
        names.clear()
        record_ftrs.clear()
        file_new = open('/sd/faceinfo.txt', 'w')
        file_new.write(''.join(''))
        file_new.close()
        img.draw_string(90, 110, b"Delete all", color=(255, 255, 255), scale=1, mono_space=1)
        lcd.display(img)

        time.sleep(2)


    a = lcd.display(img)

    del img
    gc.collect()


# a = kpu.deinit(task_fe)
# a = kpu.deinit(task_ld)
# a = kpu.deinit(task_fd)


