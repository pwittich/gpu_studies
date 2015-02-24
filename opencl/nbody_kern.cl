/* nbody_kern.cl  -*-c-*- */
/* (c) Copyright © 2009-2010 Brown Deer Technology, LLC */

__kernel void nbody_kern(
			 float dt1, float eps,
			 __global float4* pos_old,
			 __global float4* pos_new,
			 __global float4* vel,
			 __local float4* pblock) 
{
  const float4 dt = (float4){dt1,dt1,dt1,0.0f};

  int gti = get_global_id(0);
  int ti = get_local_id(0);

  int n = get_global_size(0);
  int nt = get_local_size(0);
  int nb = n/nt;
  float4 p = pos_old[gti];
  float4 v = vel[gti];

  float4 a = (float4){0.0f,0.0f,0.0f,0.0f};
  for(int jb=0; jb < nb; jb++) { /* Foreach block ... */

    pblock[ti] = pos_old[jb*nt+ti]; /* Cache ONE particle position */
    barrier(CLK_LOCAL_MEM_FENCE); /* Wait for others in the work-group */

    for(int j=0; j<nt; j++) { /* For ALL cached particle positions ... */
      // int gji = nt*ti + j; //  I think this is wrong
      // if (gji == gti ) continue;
      float4 p2 = pblock[j]; /* Read a cached particle position */
      //if ( (fabs(p2.x- p.x)<eps)&&(fabs(p2.y- p.y)<eps)&&(fabs(p2.z- p.z)<eps)  ) continue;
      float4 d = p2 - p;
      float invr = rsqrt(d.x*d.x + d.y*d.y + d.z*d.z + eps );
      float f = p2.w*invr*invr*invr; // this is actually f/(r^3 m_1), assume G = 1
      //float f = p2.w*invr;
      // d is direction btw two but not a unit vector
      // extra powers of invr above take care of length
      a += f*d; /* Accumulate acceleration */
    }

    barrier(CLK_LOCAL_MEM_FENCE); /* Wait for others in work-group */
  }
  p += dt*v + 0.5f*dt*dt*a;
  v += dt*a;

  pos_new[gti] = p;
  vel[gti] = v;

}

