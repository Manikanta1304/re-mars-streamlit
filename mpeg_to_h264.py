import cv2

########3# attention, use avc1 instead of x264 or h264
fourcc_type = 'avc1'
# fourcc_type = 'mp4v'

output_path = 'output.mp4'


def main():
    # open camera
    vc = cv2.VideoCapture('ap_only.mp4')
    if not vc.isOpened():
        print('Error: can not opencv camera')
        exit(0)

    ret, frame = vc.read()
    w = int(vc.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(vc.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = vc.get(cv2.CAP_PROP_FPS)

    fourcc = cv2.VideoWriter_fourcc(*fourcc_type)
    vw = cv2.VideoWriter(output_path, fourcc, fps, (w, h), True)
    while ret:
        vw.write(frame)
        ret, frame = vc.read()

        # cv2.imshow('frame', frame)
        # if cv2.waitKey(int(1 / fps * 1000)) & 0xFF == ord('q'):
        #     break
    # cv2.destroyAllWindows()
    vw.release()


if __name__ == '__main__':
    main()
