# 第72次 B1 2021年只读抽查

范围：仅三个现有2021年已落盘NetCDF；按文件名排序取前三个。读取变量属性、坐标和文件哈希，未读取气象变量数组，未改数据或干扰下载。
三文件均可打开，f003/006/009/012/015/018齐全，valid_time=C+f，读取前后大小和mtime一致。该检查不代替B1全量清单验收。

服务器目录：/home/XT/nuist/pythoncode/TimeDARTFirst/datasets/external/gfs_g0/dev/2021
环境：/home/XT/miniconda3/envs/gfs_g0/bin/python，xarray.open_dataset(cache=False, decode_timedelta=True)。

- gfs_2021010100_cropped.nc：215645字节，SHA256 eccaa08833cc9519d5fd5154b936a46cabcb3be9ba249820cbb2152083f65891。
- gfs_2021010106_cropped.nc：214992字节，SHA256 7b0422c70486afce33952db378cb1f7335d7372ac91d25576ad25841654c23ae。
- gfs_2021010112_cropped.nc：214448字节，SHA256 b096330c75c1c147dc57ccea2eae16650e3373f2f13828ad0f873bf3981de1a0。

三个文件的变量身份一致：
- prmsl：GRIB_paramId=260074，GRIB_typeOfLevel=meanSea，units=Pa。
- u10：GRIB_paramId=165，GRIB_typeOfLevel=heightAboveGround，units=m s**-1。
- v10：GRIB_paramId=166，GRIB_typeOfLevel=heightAboveGround，units=m s**-1。
- u100：GRIB_paramId=131，GRIB_typeOfLevel=heightAboveGround，units=m s**-1。
- v100：GRIB_paramId=132，GRIB_typeOfLevel=heightAboveGround，units=m s**-1。
- heightAboveGround为共享标量10.0；各变量没有独立GRIB_level属性。

结论：未确认“变量名u/v、没有100m”的推断；实际有u100/v100命名字段，但共享10m坐标仍不能单独证明100m来源。维持已决定的100m_used=false。
本轮不触发条件性的u/v别名适配或其合成测试；当前B2审计器对u10/v10预期131/132，不能直接作为B1正式审计器使用。下一阶段需明确支持B1的165/166身份并测试，随后才做全量验收。
本轮不读取其他年份，不判断B1已全部下载完成；等待用户通知。
