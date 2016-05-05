import paint
import cv2

if __name__ == "__main__":
    import os,sys
    dst = sys.argv[2]
    src = sys.argv[1]
    for root, dirs, filenames in os.walk(src):
        for f in filenames:
            fin = os.path.join(root,f)
            fout = os.path.join(dst,f)
            print fin
            img = cv2.imread(fin)
            warp = paint.getPaint(img,0)
            if warp is None:
                pass
            else:
                cv2.imwrite(fout,warp)
                print "output:"+fout
