src=../bin
dest=./prism/bin
project=./prism

rm -rf prism
echo "... exporting git project"
git clone https://github.com/prismplp/prism.git ./prism

rm -rf $project/.git

echo "... copying *.bin/*.exe"
cp $src/prism_up_linux.bin $dest/
cp $src/prism_mp_linux.bin $dest/
cp $src/prism_up_darwin.bin $dest/
cp $src/prism_mp_darwin.bin $dest/
cp $src/prism_win32.exe $dest/
cp $src/prism_win64.exe $dest/

echo "... copying *.out "
cp $src/prism.out $dest/
cp $src/foc.out $dest/
cp $src/batch.out $dest/
cp $src/bp.out $dest/
cp $src/mpprism.out $dest/

