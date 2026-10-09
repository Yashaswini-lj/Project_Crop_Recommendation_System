function toggleMenu(){document.getElementById("mainNav")?.classList.toggle("open");}
document.addEventListener("DOMContentLoaded",()=>{
  const form=document.getElementById("cropForm");
  const btn=document.getElementById("predictBtn");
  if(form&&btn){form.addEventListener("submit",()=>{btn.disabled=true;btn.innerHTML="⏳ Generating prediction...";});}
  const reset=document.getElementById("resetBtn");
  if(reset){reset.addEventListener("click",()=>{setTimeout(()=>window.scrollTo({top:0,behavior:"smooth"}),50);});}
});
